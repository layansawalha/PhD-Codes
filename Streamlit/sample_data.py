"""
Real sample data bundled with the demonstrator.
Study 2: five real BrEaST-Lesions-USG cases (image + clinical fields + real label).
Study 3: three real scientific PDFs for the document classifier.
Labels shown are the dataset ground truth; the tab's prediction is a transparent
surrogate, not the trained model.
"""
import os
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_assets")

def asset(name):
    return os.path.join(ASSET_DIR, name)

# ----- Study 2: real BrEaST cases (from the clinical spreadsheet) -----
STUDY2_CASES = {
    "case001 - BI-RADS 2 (benign, post-surgical scar)": {
        "image": asset("case001.png"),
        "descriptor": ("Age 57. Irregular shape with not circumscribed, indistinct margins. "
                       "Heterogeneous echogenicity with posterior shadowing. No calcifications. "
                       "BI-RADS category 2. Interpretation: breast scar following surgery and radiotherapy."),
        "birads": "2", "truth": "Benign",
    },
    "case002 - BI-RADS 4b (benign, intramammary lymph node)": {
        "image": asset("case002.png"),
        "descriptor": ("Oval shape with not circumscribed, indistinct margins. Hypoechoic, no posterior "
                       "features, no calcifications. BI-RADS category 4b. Interpretation: dysplasia versus "
                       "fibroadenoma. Final diagnosis: intramammary lymph node."),
        "birads": "4b", "truth": "Benign",
    },
    "case003 - BI-RADS 4a (benign, ductal hyperplasia)": {
        "image": asset("case003.png"),
        "descriptor": ("Age 56. Oval shape with circumscribed margins. Hyperechoic, no posterior features, "
                       "no calcifications. BI-RADS category 4a. Interpretation: duct filled with thick fluid "
                       "versus intraductal papilloma."),
        "birads": "4a", "truth": "Benign",
    },
    "case004 - BI-RADS 3 (benign, cyst)": {
        "image": asset("case004.png"),
        "descriptor": ("Age 43. Round shape with circumscribed margins. Hypoechoic, no posterior features, "
                       "no calcifications. BI-RADS category 3. Interpretation: cyst filled with thick fluid."),
        "birads": "3", "truth": "Benign",
    },
    "case005 - BI-RADS 4b (malignant, papillary carcinoma)": {
        "image": asset("case005.png"),
        "descriptor": ("Age 67. Oval shape with circumscribed margins. Complex cystic and solid echogenicity "
                       "with posterior enhancement. No calcifications. BI-RADS category 4b. Interpretation: "
                       "suspicion of malignancy. Final diagnosis: encapsulated papillary carcinoma with DCIS."),
        "birads": "4b", "truth": "Malignant",
    },
}

# ----- Study 3: three real scientific PDFs -----
STUDY3_PDFS = {
    "Ramsey graphs / dispersers (theory)": asset("1.pdf"),
    "Information security case study (applications)": asset("2.pdf"),
    "Reflective meta-reasoning / type theory (methods)": asset("3.pdf"),
}
