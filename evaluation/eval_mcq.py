from chatgptinterface import ChatGPTInteface
import json
import numpy as np
import pandas as pd
import os

def calculate_jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    
    jaccard_similarity = intersection / union
    return jaccard_similarity

def are_strings_similar(str1, str2, threshold=0.75):
    similarity = calculate_jaccard_similarity(str1, str2)
    return similarity >= threshold

ORGANIZATION = os.getenv("ORGANIZATION")
API_KEY = os.getenv("API_KEY")
save_path = "datas"

chatGPTInteface = ChatGPTInteface(API_KEY=API_KEY, organization=ORGANIZATION)
interface = ChatGPTInteface(API_KEY, ORGANIZATION, model_name="gpt-3.5-turbo")
json_list = sorted(os.listdir(save_path))

counts = []
logs = []
for json_item in json_list:
    gt_json = json.load(open(os.path.join(save_path, json_item, "scene", "answer_gt.json")))
    pred_json = json.load(open(os.path.join(save_path, json_item, "scene", "answer_pred_both.json")))
    for i in gt_json:
        del i["matched_coords"]
        del i["annotation"]

    for i in pred_json:
        del i["matched_coords"]
        del i["annotation"]

    question_types = ["SPATIAL_REASONING", "INSTANCE_ATTRIBUTE", "INSTANCE_COUNTING", "VISUAL_REASONING"]
    for qntype in question_types:
        try:
            response = interface.generate_question(gt_json, question_type=qntype)
            question_with_answer = response["choices"][0]["message"]["content"]
            print("response \n", question_with_answer)
            print("\n\n")
            answer_ind = response["choices"][0]["message"]["content"].lower().find("answer")
            generated_question = question_with_answer[:answer_ind]
            correct_answer = question_with_answer[answer_ind+8:]
            print("separated question \n", generated_question)
            print("\n\n")
            print("separated answer \n", correct_answer)
            response = interface.generate_conversation(pred_json, generated_question, conversation_type="MCQ")
            chatgpt_answer = response["choices"][0]["message"]["content"]
            print("\n\n\n")
            print("selected answer \n", chatgpt_answer)
            count = are_strings_similar(chatgpt_answer, correct_answer)
            print(count)
            counts.append(count)
            logs.append([
                generated_question,
                correct_answer,
                chatgpt_answer
            ])
        except Exception as e:
            print(e)
            pass

counts = np.array(counts)
print("MCQ Accuracy: \t", (counts.sum())/len(counts))
df = pd.DataFrame( data=logs, columns=["Question", "Correct Answer", "ChatGPT Answer"])
df.to_csv("logs.csv")
