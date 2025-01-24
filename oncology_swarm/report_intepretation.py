import os
from swarms import Agent
from swarm_models import OpenAIChat

# Set up OpenAI model
llm = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=4000,
    model_name="gpt-4o",
)

# Define the 5 specialized agents
medical_terms_agent = Agent(
    agent_name="Medical Terms Agent",
    system_prompt="Given the pathology report, explain medical terms in the pathology report using simple, everyday language.",
    llm=llm,
    max_loops=1,
)

understanding_report_agent = Agent(
    agent_name="Understanding Report Agent",
    system_prompt="Given the pathology report, explain the pathology report in detail using simple language.",
    llm=llm,
    max_loops=1,
)

def run(pathology_report):
    # Example pathology report
    pathology_report = """
    PATIENT HISTORY: The patient is a. with a serum PSA level of 7.4 ng/ml. He has a family history of prostate cancer. The patient also nas a history OF. in. S. The patient had and outside prostate biopsy (not reviewed at. reported as. prostatic adenocarcinoma: left apex (Gleason score 3+3=6), left mid (Gleason score 3+4=7), left base (Gleason score 3+3=6), right. apex (Gleason score 3+3=6), right mid (Gleason score 3+3=6), and right base (Gleason score 3+3=6). The clinical stage is T1c. PRE-OP DIAGNOSIS: Prostate cancer. POST-OP DIAGNOSIS: Same. PROCEDURE: Radical prostatectomy. FINAL DIAGNOSIS: PART 1: RIGHT PELVIC LYMPH NODES, EXCISION -. FOUR BENIGN LYMPH NODES (0/4). NO EVIDENCE OF MALIGNANCY. PART 3: LEFT PELVIC LYMPH NODES, EXCISION -. THIRTEEN BENIGN LYMPH NODES (0/13). NO EVIDENCE OF MALIGNANCY. PART 3: PROSTATE, BILATERAL SEMINAL VESICLES AND BILATERAL DISTAL VASA DEFERENTIA, RADICAL. PROSTATECTOMY -. A. PROSTATIC ADENOCARCINOMA, ACINAR TYPE, GLEASON SCORE 3+4=7, WITH A TERTIARY FOCUS OF. GLEASON PATTERN 5 PROSTATIC ADENOCARCINOMA (See comment). B. CARCINOMA INVOLVES BOTH RIGHT AND LEFT LOBES OF THE PROSTATE GLAND AND HAS A MAXIMAL. TUMOR DIAMETER OF 2 CM IN A HISTOLOGIC SECTION. C. CARCINOMA COMPRISES APPROXIMATELY 15% OF THE SAMPLED PROSTATE GLAND VOLUME. D. FOCAL CARCINOMATOUS EXTRACAPSULAR EXTENSION IS PRESENT IN THE RIGHT POSTERIOR MID AND. RIGHT POSTERIOR BASE PROSTATE (SECTIONS 3V, 3WW). E. MULTIFOCAL CARCINOMATOUS PERINEURAL INVASION IS PRESENT. F. NO ANGIOLYMPHATIC INVASION IS PRESENT. G. MULTIFOCAL HIGH-GRADE PROSTATIC INTRAEPITHELIAL NEOPLASIA. H. BENIGN SEMINAL VESICLES AND DISTAL VASA DEFERENTIA. I. ALL SURGICAL RESECTION MARGINS ARE FREE OF NEOPLASIA. J. NON-NEOPLASTIC PROSTATE WITH MILD NODULAR HYPERPLASIA AND FOCAL GLANDULAR ATROPHY. K. TNM PATHOLOGIC STAGE: T3a NO MX. "STOLOGIC GRADE: G3-4 (See synoptic). COMMENT: Part 3: There is a small (0.3 cm) focus of Gleason pattern 5 carcinoma present in the left mid anterior-posterior prostate. (slides 30, 3V). This likely represents a small focus of dedifferentiation of the prostatic adenocarcinoma. Gleason 4 and. 5 components comprise approximately 20% of the total sampled tumor volume. CASE SYNOPSIS: SYNOPTIC DATA - PRIMARY PROSTATE TUMORS. CLINICAL DATA: PSA value: 7.4. INVASIVE CA IDENTIFIED?: TUMOR HISTOLOGY: Adenocarcinoma NOS. PRIMARY GLEASON GRADE: 3. SECONDARY GLEASON GRADE: 4. GLEASON SUM SCORE: 7. GLEASON 4/5 PERCENTAGE: 20%. WEIGHT OF PROSTATE: 54.97gm. TUMOR SIZE: Maximum dimension: 2 cm. LOBE LATERALITY: Right and Left Lobes. PERCENT OF SPECIMEN INVOLVED BY TUMOR: 5 25%. MULTIFOCAL DISEASE: HIGH GRADE PIN: Yes - multifocal. EXTRAPROSTATIC EXTENSION: Yes Focal. PERINEURAL INVASION: ANGIOLYMPHATIC INVASION: SEMINAL VESICLE INVASION: SURGICAL MARGIN INVOLVEMENT: All surgical margins free of tumor. LYMPH NODES EXAMINED: 17. LYMPH NODES POSITIVE: 0. T STAGE, PATHOLOGIC: pT3a. N STAGE, PATHOLOGIC: pNo. M STAGE, PATHOLOGIC: pMX. HISTOLOGIC GRADE: G3-4, Poorly differentiated/undifferentiated. Comment: There is a n 3 rm farne of Clannan.
    """


    # Run the agents independently
    medical_terms_output = medical_terms_agent.run(
        f"""
        Here is the pathology report: {pathology_report}
        Perform the following tasks:
        1. Explain the medical terms.
        """
    )
    understanding_report_output = understanding_report_agent.run(pathology_report)

    # Clean the outputs (remove unnecessary metadata)
    def clean_output(output):
        if "Output:" in output:
            return output.split("Output:")[1].strip()
        return output

    medical_terms_output_cleaned = clean_output(medical_terms_output)
    understanding_report_output_cleaned = clean_output(understanding_report_output)

    # Combine the cleaned outputs into a single string
    final_output = (
        "1. Explanation of Medical Terms:\n" + medical_terms_output_cleaned + "\n\n" +
        "2. Detailed Explanation of the Report:\n" + understanding_report_output_cleaned
    )
    return final_output