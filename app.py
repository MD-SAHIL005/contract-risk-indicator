import os
import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Contract Risk Indicator", page_icon="📄", layout="wide")
st.title("📄 AI-Based Contract Risk Indicator")
st.write("Upload a contract to classify clauses with a trained text model and check for potentially missing information.")
st.info("This is an educational prototype, not legal advice. Model categories and confidence scores are not validated legal risk scores.")

MODEL_FILE = "contract_risk_model.joblib"

RISK_RULES = {
    "Automatic Renewal": {
        "patterns": [r"\bautomatically renew(?:s|ed|ing)?\b", r"\bautomatic renewal\b", r"\bagreement renews automatically\b", r"\bsuccessive periods of \d+ months\b"],
        "priority": "Medium",
        "explanation": "Review the renewal period, notice deadline, and how a party can prevent renewal."
    },
    "Termination": {
        "patterns": [r"\bterminate this agreement\b", r"\bearly termination\b", r"\btermination without notice\b", r"\bimmediate termination\b", r"\bsuspend services immediately\b", r"\bmay terminate\b"],
        "priority": "Medium",
        "explanation": "Review termination or suspension conditions, notice periods, and each party's rights."
    },
    "Penalties": {
        "patterns": [r"\bpenalt(?:y|ies)\b", r"\bcancellation fee\b", r"\btermination fee\b", r"\blate fee\b", r"\bliquidated damages\b"],
        "priority": "High",
        "explanation": "Review the amount, calculation, and circumstances in which the charge applies."
    },
    "Liability": {
        "patterns": [r"\bliability\b", r"\bliable for all damages\b", r"\bindemnif(?:y|ication)\b", r"\bconsequential losses\b", r"\bdamages arising\b"],
        "priority": "High",
        "explanation": "Review the scope of liability, who bears responsibility, and whether limits or exceptions apply."
    },
    "Payment Terms": {
        "patterns": [r"\bpayment is due\b", r"\bpayments? shall be made\b", r"\binvoices? (?:must|shall|will) be paid\b", r"\bwithin \d+ days of (?:the )?receipt\b", r"\blate payment\b", r"\binterest on overdue payments\b", r"\bpayment deadline\b"],
        "priority": "Informational",
        "explanation": "Check payment deadlines, amounts, invoicing process, and conditions for payment."
    },
    "Confidentiality": {
        "patterns": [r"\bconfidential(?:ity)?\b", r"\btrade secrets\b", r"\bnon-disclosure\b"],
        "priority": "Medium",
        "explanation": "Review what information is protected, permitted disclosures, and how long the obligation lasts."
    }
}

MISSING_CHECKS = {
    "Effective date": {
        "present": [r"\beffective date\b", r"\beffective as of\b", r"\bdated this\b", r"\bdate of agreement\b", r"\bcommencement date\b", r"\bentered into on\b"],
        "missing": [r"\bnot entered an effective date\b", r"\beffective date.{0,50}\b(?:missing|not provided|not entered|omitted)\b", r"\bno effective date\b"],
        "why": "The effective date helps identify when the agreement starts.",
        "action": "Confirm and insert the agreed effective date."
    },
    "Notice address": {
        "present": [r"\bnotice address\b", r"\bnotices shall be sent\b", r"\baddress for notices\b", r"\baddress for service\b", r"\bsend written notice to\b"],
        "missing": [r"\bnot entered.{0,80}\bnotice address\b", r"\bnotice address.{0,50}\b(?:missing|not provided|not entered|omitted)\b", r"\bno notice address\b"],
        "why": "A clear notice address helps parties know where to send formal notices.",
        "action": "Confirm and insert the correct address or agreed notice method."
    },
    "Named contact": {
        "present": [r"\bcontact person\b", r"\bnamed representative\b", r"\bnamed contact\b", r"\bcontact details\b", r"\bdesignated representative\b", r"\bfor the attention of\b", r"\baccount manager\b"],
        "missing": [r"\bnot entered.{0,80}\bnamed contact\b", r"\bno named contact\b", r"\bnamed contact.{0,50}\b(?:missing|not provided|not entered|omitted)\b", r"\bcontact details.{0,50}\b(?:missing|not provided|not entered|omitted)\b"],
        "why": "A named contact can clarify who handles formal communications.",
        "action": "Confirm whether a responsible contact and contact details should be added."
    },
    "Payment deadline": {
        "present": [r"\bwithin \d+ days\b", r"\bpayment due\b", r"\bdue date\b", r"\bpayable within\b", r"\bnet\s*\d+\b", r"\binvoices? (?:must|shall|will) be paid\b"],
        "missing": [r"\bpayment deadline.{0,50}\b(?:missing|not provided|not entered|omitted)\b", r"\bno payment deadline\b", r"\bpayment due date.{0,50}\b(?:missing|not provided|not entered|omitted)\b"],
        "why": "A clear payment deadline helps both parties understand when payment is due.",
        "action": "Confirm that the payment due date or calculation method is clearly stated."
    }
}

HEADINGS = {
    "automatic renewal", "termination", "early termination fee", "payment terms",
    "liability", "penalties", "parties and services", "service agreement",
    "missing details to confirm", "termination by provider", "confidentiality"
}

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_text(uploaded_file):
    raw = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".pdf"):
        import pymupdf
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            return "\n".join(page.get_text() for page in document)
    return raw.decode("utf-8-sig", errors="replace")

def check_missing_information(contract_text):
    text = normalize_whitespace(contract_text)
    missing_items = []
    for item, config in MISSING_CHECKS.items():
        explicitly_missing = any(re.search(p, text, re.IGNORECASE) for p in config["missing"])
        found = any(re.search(p, text, re.IGNORECASE) for p in config["present"])
        if explicitly_missing or not found:
            missing_items.append(item)
    return missing_items

def split_clauses(text):
    """
    Split by paragraphs/numbered clauses, not every visual PDF line or sentence.
    PDF extractors often add line breaks in the middle of a sentence.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]

    blocks = []
    current_lines = []

    clause_start = re.compile(
        r"^(?:(?:section|clause|article)\s+\d+(?:\.\d+)*[.): -]*|"
        r"\d+(?:\.\d+){0,4}[.)]?\s+|(?:[A-Z][.)])\s+)",
        re.IGNORECASE
    )

    def flush():
        nonlocal current_lines
        if current_lines:
            block = normalize_whitespace(" ".join(current_lines))
            if block:
                blocks.append(block)
            current_lines = []

    for line in lines:
        if not line:
            flush()
            continue

        clean_heading = line.rstrip(":").strip().casefold()
        is_heading = clean_heading in HEADINGS
        is_numbered_start = bool(clause_start.match(line))

        if is_heading or is_numbered_start:
            flush()
            if not is_heading:
                current_lines.append(line)
            continue

        current_lines.append(line)

    flush()

    filtered = []
    seen = set()
    for block in blocks:
        cleaned = block.strip()
        if cleaned.rstrip(":").casefold() in HEADINGS or len(cleaned) < 35:
            continue
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            filtered.append(cleaned)

    return filtered

def classify_clauses(contract_text, model):
    results = []
    for clause in split_clauses(contract_text):
        if len(clause) < 20:
            continue
        category = str(model.predict([clause])[0])
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba([clause])[0]))
        rule = RISK_RULES.get(category)
        priority = rule["priority"] if rule else ("Informational" if category == "Other" else "Review")
        explanation = rule["explanation"] if rule else (
            "The model did not match this clause to a configured review category. Check it manually."
        )
        review_flag = (
            "Needs manual review"
            if confidence is None or confidence < 0.40
            else "Check prediction"
        )
        results.append({
            "Category": category,
            "Priority": priority,
            "Model confidence": f"{confidence:.0%}" if confidence is not None else "Not available",
            "Review flag": review_flag,
            "Evidence": clause,
            "Explanation": explanation
        })
    return results

uploaded_file = st.file_uploader(
    "Upload a contract", type=["pdf", "txt"],
    help="Text-based PDFs and UTF-8 TXT files are supported. Scanned PDFs may require OCR."
)

if not os.path.exists(MODEL_FILE):
    st.warning(
        f"Trained model file `{MODEL_FILE}` was not found. First run `python train_model.py` "
        "in the same folder to create it, then refresh this app."
    )
else:
    try:
        model = joblib.load(MODEL_FILE)
        st.success("Trained clause-classification model loaded.")
    except Exception as error:
        model = None
        st.error(f"Could not load the trained model: {error}")

if uploaded_file is not None and os.path.exists(MODEL_FILE):
    st.success(f"Uploaded: {uploaded_file.name}")
    if st.button("Analyse Contract", type="primary"):
        try:
            text = extract_text(uploaded_file)
            if not text.strip():
                st.error("No text could be extracted. This may be a scanned PDF that requires OCR.")
                st.session_state.pop("analysis_result", None)
            elif model is None:
                st.error("The model could not be loaded. Check the error above.")
            else:
                st.session_state["analysis_result"] = {
                    "text": text,
                    "findings": classify_clauses(text, model),
                    "missing": check_missing_information(text),
                    "filename": uploaded_file.name
                }
        except Exception as error:
            st.error(f"Document processing failed: {error}")

if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    findings = result["findings"]
    missing_items = result["missing"]
    text = result["text"]

    st.divider()
    st.header("📊 Contract Analysis Dashboard")
    high = sum(x["Priority"] == "High" for x in findings)
    medium = sum(x["Priority"] == "Medium" for x in findings)
    c1, c2, c3 = st.columns(3)
    c1.metric("Classified Clauses", len(findings))
    c2.metric("High-Priority Review Categories", high)
    c3.metric("Medium-Priority Review Categories", medium)
    st.caption("Priority is assigned from category-level rules. Model confidence is not a probability that a clause is legally risky.")

    with st.expander("📄 View Original Contract Text"):
        st.text_area("Extracted text", text, height=250, key="original_contract_text")

    st.subheader("🔍 ML-Classified Clauses")
    if findings:
        st.dataframe(pd.DataFrame([
            {"Category": x["Category"], "Priority": x["Priority"],
             "Model confidence": x["Model confidence"], "Review flag": x["Review flag"],
             "Evidence": x["Evidence"]}
            for x in findings
        ]), width="stretch", hide_index=True)

        st.subheader("📝 Human Review")
        for i, finding in enumerate(findings):
            with st.expander(f"{i + 1}. {finding['Category']} — {finding['Priority']}"):
                st.markdown("**Original evidence**")
                st.write(finding["Evidence"])
                st.markdown("**Why review it?**")
                st.write(finding["Explanation"])
                st.caption(f"Model confidence: {finding['Model confidence']} · {finding['Review flag']}")
                st.selectbox(
                    "Review status", ["Not reviewed", "Relevant", "Not relevant"],
                    key=f"status_{result['filename']}_{i}"
                )
                st.text_input("Reviewer notes", key=f"notes_{result['filename']}_{i}")
    else:
        st.warning("No sufficiently long clauses were available to classify. Check the extracted text.")

    st.divider()
    st.subheader("📋 Possible Missing Information")
    if missing_items:
        st.warning("These items were explicitly described as missing or were not detected by the configured checks. Verify them against the full contract.")
        for item in missing_items:
            cfg = MISSING_CHECKS[item]
            with st.container(border=True):
                st.markdown(f"**{item}**")
                st.caption(f"Why it matters: {cfg['why']}")
                st.write(f"Suggested action: {cfg['action']}")
    else:
        st.success("All configured information checks found matching phrases. This does not prove the contract is complete; verify manually.")

    if findings:
        csv_data = pd.DataFrame(findings).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Findings as CSV", data=csv_data,
            file_name="contract_risk_report.csv", mime="text/csv"
        )

    st.caption("This educational prototype classifies clause categories using a small demonstration dataset. It is not a validated legal-risk model. Have a qualified reviewer check the full agreement.")
