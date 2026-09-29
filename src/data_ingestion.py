import json
import random
import sys
import os
import pandas as pd
from utils import logger


logger = logger.get_logger(__name__)

LABELS = [
    "repetition",
    "unclear_intro",
    "missing_confirmation",
    "long_agent_monologue",
    "compliance_risk"
]


#Splits a conversation into INTRO / BODY / CLOSING to help the model localize call-quality issues.
def conversation_to_structured_text(conversation):
    intro, body, closing = [], [], []
    agent_turns = 0
    customer_turns = 0
    agent_intro_present = False
    confirmation_asked = False

    for turn in conversation:
        speaker = turn.get("speaker", "").lower()
        text = turn.get("text", "").strip()

        if not text:
            continue

        if speaker == "agent":
            agent_turns += 1
            if agent_turns == 1:
                agent_intro_present = "name" in text.lower()
            if "confirm" in text.lower():
                confirmation_asked = True
        else:
            customer_turns += 1

        line = f"{speaker.capitalize()}: {text}"

        if agent_turns <= 2:
            intro.append(line)
        elif agent_turns <= 6:
            body.append(line)
        else:
            closing.append(line)

    signal_block = (
        "[META]\n"
        f"Agent_intro_present: {agent_intro_present}\n"
        f"Agent_turns: {agent_turns}\n"
        f"Customer_turns: {customer_turns}\n"
        f"Confirmation_asked: {confirmation_asked}\n\n"
    )

    return (
        signal_block +
        "[INTRO]\n" + "\n".join(intro) +
        "\n\n[BODY]\n" + "\n".join(body) +
        "\n\n[CLOSING]\n" + "\n".join(closing)
    )



class DataIngestion:

    def __init__(self):
        pass

    # read the train file data
    @staticmethod
    def _read_train_data_file(file_path: str):
        try: 
            if not os.path.exists(file_path):
                logger.error("File path does not exist : %s" , file_path)
                return {
                    "error": f"File Path not found : {file_path}" , 
                }

            json_data = []

            with open(file_path, "r") as f:
                json_data = json.load(f)
            logger.info("Successfully data read of file : %s" , file_path.split("/")[-1])
            return json_data


        except Exception as e:
            exc_tb = sys.exc_info()[2]
            logger.error(
                "Failed To get data error is : %s in line no : %s",
                str(e),
                exc_tb.tb_lineno
            )
            return {'error': str(e)}


    @staticmethod
    # extract some sample data
    def extract_sample_data(raw_data  , output_file_path):

        if raw_data and isinstance(raw_data , list):
            df = pd.DataFrame(raw_data)
            df_first_2000 = df.iloc[:2000]
           

            # SAVE SAMPLE DATA
            df_first_2000.to_json(
                output_file_path,
                orient="records",
                indent=2
            )
           
            
            return df_first_2000

        return []