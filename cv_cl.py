from pprint import pprint
import shutil
import random, os
from tqdm import tqdm

from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
load_dotenv()



JOB_POSITIONS_PATH = "/Users/alex/Desktop/programmazione/notebooks/My notebooks/Rizzo AI Academy/Agenti in Python/CV_modifier/job_positions"

job_descriptions = []

for job_position in os.listdir(JOB_POSITIONS_PATH):
    if job_position.startswith('.'):
        continue
    JOB_POSITION_PATH = os.path.join(JOB_POSITIONS_PATH, job_position)
    with open(JOB_POSITION_PATH) as file:
        try:
            job_description = file.read()
            if len(job_description) == 0:
                continue
            job_descriptions.append(f"\nJOB DESCRIPTION:\n\n{job_description}")
        except:
            print(f"Error reading file: {JOB_POSITION_PATH}")
       

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.media import File
from agno.tools.gmail import GmailTools

CV_PATH = "/Users/alex/Desktop/cose per concorsi/curriculum/Curriculum -- Alessio Fontanarossa.pdf"
CL_PATH = "/Users/alex/Desktop/cose per concorsi/curriculum/Cover Letter - Alessio Fontanarossa.pdf"
cv_pdf = File(filepath = CV_PATH)
cl_pdf = File(filepath = CL_PATH)

id = "gpt-4o-mini"

class CurriculumVitae(BaseModel):
    agency_name: str = Field(..., description = "Name of the agency")
    job_position: str = Field(..., description = "Name of the job position (es. 'Data Scientist')")
    first_cv_version: str = Field(..., description = "First version of the 'Summary' section") 
    second_cv_version: str = Field(..., description = "Second version of the 'Summary' section")
    third_cv_version: str = Field(..., description = "Third version of the 'Summary' section")
class CoverLetter(BaseModel):
    first_cl_version: str = Field(..., description = "First version of the cover letter") 
    second_cl_version: str = Field(..., description = "Second version of the cover letter")

cv_agent = Agent(
    name = "cv_agent",
    model = OpenAIChat(id = id),
    description = """You are an HR manager which helps the user to write better its curriculum vitae.
    You ALWAYS remember that the user has over 4 years of experience in Black Hole Physics, but he is looking for a job in the tech sector, especially in Data Science, Machine Learning and AI Engineering.
    The user has over 1 year of experience in the AI sector, not 4 years.
    You will receive a job description (both in Italian or English) in the tech sector (Data Science, Machine Learning, AI Engineering and so on). 
    After a scrupulous analysis of the text you will catch the core messages, values of the agency and keywords.""",
    instructions = ["1) Read the job description, taking the key insights.",
                    "2) Read the Alessio's curriculum, especially the content of the 'Summary' section.",
                    """3) Produce a new 'Summary' section, combining the original informations of Alessio's cv and the keywords of the job description.
                    DO NOT use the name of the agency.""",
                    "4) ALWAYS use the hard and soft skills of the original Alessio'curriculum.",
                    "5) DO NOT mention SQL skills."],
    expected_output = "Name of the agency, name of the job position and 3 versions of the same 'Summary' section ALWAYS in english, with approximately the same number of words of the original 'Summary' section.",
    output_schema = CurriculumVitae)

cl_agent = Agent(
    name = "cl_agent",
    model = OpenAIChat(id = id),
    description = """You are an HR manager which helps the user to write better its cover letter.
    You ALWAYS remember that the user has over 4 years of experience in Black Hole Physics, but he is looking for a job in the tech sector, especially in Data Science, Machine Learning and AI Engineering.
    The user has over 1 year of experience in the AI sector, not 4 years.
    You will receive a job description (both in Italian or English) in the tech sector (Data Science, Machine Learning, AI Engineering and so on). 
    After a scrupulous analysis of the text you will catch the core messages, values of the agency and keywords.""",
    instructions = ["1) Read the job description, taking the key insights.",
                    "2) Read the Alessio's cover letter, especially the first and last paragraph of the main body.",
                    """3) Produce a new cover letter, combining the original informations of Alessio's cover letter and the key insights of the job description.""",
                    "4) ALWAYS use the name of the agency and refer to the job position name, showing interest in the position.",
                    "5) DO NOT mention SQL skills.",
                    "6) DO NOT change the structure of the original cover letter.",],
    expected_output = "2 versions of the same cover letter ALWAYS in english, with approximately the same number of words of the original cover letter.",
    output_schema = CoverLetter)

email_agent = Agent(
    name = "cl_agent",
    model = OpenAIChat(id = id),
    description = """You are a useful assistant that send email.
    You will receive the name of an agency, the name of the job position, three versions of the same currivulum vitae 
    and two versions of the same cover letter.""",
    instructions = ["1) Write an email to 'alessio.font7@gmail.com', with subject the 'name of the agency - job position' you received",
                    """2) Write in the body of the emai exactly the following content:
                        'Prima versione del cv':\n\n first version of the cv\n\n,
                        'Seconda versione del cv':\n\n second version of the cv\n\n,
                        'Terza versione del cv':\n\n third version of the cv\n\n,
                        'Prima versione della cl':\n\n first version of the cover letter\n\n,
                        'Seconda versione della cl':\n\n second version of the cover letter""",
                    "3) For what concerns the cover letter, write in the email ONLY the main body "],
    expected_output = "A well written email, with titles in markdown format.",
    tools = [GmailTools(port = 8080)])

for job_description in job_descriptions:
    cv_output = cv_agent.run(job_description, files = [cv_pdf])

    AGENCY_NAME = cv_output.content.agency_name
    JOB_POSITION = cv_output.content.job_position
    FIRST_CV_VERSION = cv_output.content.first_cv_version
    SECOND_CV_VERSION = cv_output.content.second_cv_version
    THIRD_CV_VERSION = cv_output.content.third_cv_version

    print(f"CV for {AGENCY_NAME} - {JOB_POSITION} created.")

    cl_output = cl_agent.run(job_description, files = [cv_pdf])

    print(f"Cover Letter for {AGENCY_NAME} - {JOB_POSITION} created.")

    FIRST_CL_VERSION = cl_output.content.first_cl_version
    SECOND_CL_VERSION = cl_output.content.second_cl_version

    send_email_prompt = f"""
    Send an email to 'alessio.font7@gmail.com' with the following informations:
    - name of the agency is {AGENCY_NAME}
    - name of the job position is {JOB_POSITION}
    - first version of the curriculum {FIRST_CV_VERSION}
    - second version of the curriculum {SECOND_CV_VERSION}
    - third version of the curriculum {THIRD_CV_VERSION}
    - first version of the cover letter {FIRST_CL_VERSION}
    - second version of the cover letter {SECOND_CL_VERSION}
    REPORT ONLY the main body of the cover letter versions (es: from 'dear hiring manager' to 'best regrards').
    """

    final = email_agent.run(send_email_prompt, markdown = True)
    print(final.content)

    NEW_PATH = f"/Users/alex/Desktop/cose per concorsi/curriculum/{AGENCY_NAME}-{JOB_POSITION}"
    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH)

