import openai
import os

# chat gpt credentials
ORGANIZATION = os.getenv("ORGANIZATION")
API_KEY = os.getenv("API_KEY")

# question types to ask
QUESTION_TYPES = ["SPATIAL_REASONING", "INSTANCE_ATTRIBUTE", "INSTANCE_COUNTING", "VISUAL_REASONING"]

# to generate per type
NUM_PER_TYPE = 10

JSONS_LIMIT = 1000

# example conversation to prompt GPT
EXAMPLE_MCQ = "Here is an example\
    What is our planet's name?\
        A. Earth\
        B. Sun\
        C. Mars\
        D. Jupyter\
        \n\n \
        Answer: A. Earth"

# scene data format
SCENE_FORMAT = {
	"bev_centroid": "2D coordinates of the center of the object",
	"bev_area": "2D area of the object",
  "brief_label": "A brief caption describing the object.",
  "bg_description": "A description of the background around the object.",
  "weather": "Weather description of the scene"
}

# question context
QUESTION_CONTEXT = {
  "SPATIAL_REASONING": "The questions should be about spatial relations between two objects. The question should be mainly based on the coordinates of the two objects. To answer the question, one should find the two mentioned objects, and find their relative spatial relation to answer the question.",
  "INSTANCE_ATTRIBUTE": "The questions should be about the attribute of a certain object, such as its color, shape or fine-grained type. Do not use the object ID in the question.",
  "INSTANCE_COUNTING": "The questions should involve the number of appearance of a certain object. Start with 'How many ....'. The choices of the question should be numbers. To answer the question, one should find and count all of the mentioned objects in the image. Make sure the options are far apart.",
  "VISUAL_REASONING": "Create complex questions beyond describing the scene. The questions should aid the user who is driving the ego-vehicle. To answer such question, one should first understanding the visual content, then based on the background knowledge or reasoning, either explain why the things are happening that way, or provide guides and help to user's request. Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first. Do not use the object ID in the question. Do not mention the coordinates."
}

# question generation format
QUESTION_FORMAT = {
   "question_number": "Please give the question number of this question.",
   "question": "Please give the question.",
   "options": "this should be a list of options",
   "correct_option": "this should be the correct options",
}

# answer format - spatial operators
ANSWER_FORMAT = {
   "inferred_query": "your interpretation of the user query in a succinct form",
   "relevant_objects": "python list of all relevant object ids for the user query",
   "query_achievable": "whether or not the user-specified query is achievable using the objects and descriptions provided in the scene.",
   "spatial_reasoning_functions": "If the query needs calling one or more spatial reasoning functions, this field contains a list of function calls that conform to the API above. Else, this field contains an empty list.",
   "explanation": "A brief explanation of what the relevant objects, and how it addresses the task."
}

# spatial operators
SPATIAL_OPERATORS_LIST = [
	"filter_front(list_of_objects, object_id) : Within the the list of objects, it returns the list of objects to front of the object with id as object_id",
	"filter_left(list_of_objects, object_id) : Within the the list of objects, it returns the list of objects to left of the object with id as object_id",
	"filter_right(list_of_objects, object_id) : Within the the list of objects, it returns the list of objects to right of the object with id as object_id",
	"filter_rear(list_of_objects, object_id) : Within the the list of objects, it returns the list of objects to rear of the object with id as object_id",
	"find_distance(list_of_objects, object_id_1, object_id_2) : Within the the list of objects, it returns the distance within 2 objects object_id_1 and object_id_2",
	"find_objects_within_distance(list_of_objects, object_id, distance) : Within the the list of objects, it returns the list of objects within distance to the object with id object_id",
    "get_k_closest_objects(list_of_objects, object_id, k) : Within the the list of objects, it returns the list of k closest objects to the object with id object_id",
    "get_k_farthest_objects(list_of_objects, object_id, k) : Within the the list of objects, it gets the k farthest objects to the object with id object_id",
    "filter_objects_with_tag(list_of_objects, object_id, tagname, d) : Within the the list of objects, it finds the objects which have tag as tagname and are within d distance to the object with id object_id",
    "filter_color(list_of_objects, object_id, colorname, d): Within the the list of objects, itfinds the objects which have color as colorname and are within d distance to the object with id object_id",
    "filter_size(list_of_objects, object_id, distance, min_size, max_size): Within the the list of objects, itfinds the objects which have size between min_size and max_size and are within d distance to the object with id object_id"
]

def setup_question_generation(question_type):
    message = f"You will be given, as input a 2D road scene in Bird's Eye View, as a list. The ego-vehicle is at (100, 100) facing along the positive Y-axis. Each entry in the list describes one object in the scene, with the following five fields: \
    \n{str(SCENE_FORMAT)}\n\
    Once you have parsed the JSON and are ready to generate question about the scene, Create {NUM_PER_TYPE} distinct multi-choice question about the scene, and provide the choices and answer. Each question should have 4 options, out of which only 1 should be correct. Do not use the object ID in the question. Do not mention the coordinates."# + "You have to return a list of JSONs each containing an MCQ question."
    message += QUESTION_CONTEXT[question_type]
    return {
        "role": "system",
        "content": message  + "Please provide answer as well. \n" + str(EXAMPLE_MCQ) + "\n . NOTE: DO NOT ask simple questions like 'What is the central object in the scene?', or What is the color of the car?." + "\n\n. "
        # "content": message  + "Please provide answer as well. \n" + "\n . NOTE: DO NOT ask simple questions like 'What is the central object in the scene?'." + "\n\n. For each question, your JSON should contain - " + str(QUESTION_FORMAT) + "\n. Your final output should be a list of JSONs, and each string should be in double quotes. Please ensure your list can be parsed in a python program. Your string should be in double quotes. Please ensure this."
    }

def add_conversation_context(conversation_type):
    if conversation_type == "MCQ":
        return  "The user will then begin to ask Multiple Choice Questions, and the task is to answer various user queries about the scene. For each question, answer with just one correct option. You should only output the correct answer. Do not give any explanation"

    if conversation_type == "SPATIAL":
        return  f"The user will then begin to ask questions, and the task is to answer various user queries about the scene. These questions will involve spatial reasoning. To assist with such queries, we have the following available functions:.\
                    {str(SPATIAL_OPERATORS_LIST)}\
                    For each user question, respond with a JSON dictionary with the following fields:\
                    {str(ANSWER_FORMAT)}\
                    The Object ID of ego-vehicle is 0. Only output this JSON. Do not output any explanation."

def setup_conversation(conversation_type="MCQ"):
    message = f"You will be given, as input a 2D road scene in Bird's Eye View, as a list. The ego-vehicle is at (0,0) facing along the positive Y-axis. Each entry in the list describes one object in the scene, with the following five fields: \
                    \n\n {str(SCENE_FORMAT)}  \n\n  \
                Once you have parsed the JSON and are ready to answer questions about the scene, please wait for the user to input JSON."
    message += add_conversation_context(conversation_type)
    return {
        "role": "system",
        "content": message
    }

class ChatGPTInteface:
    def __init__(self, API_KEY, organization, model_name="gpt-4") -> None:
        openai.api_key = API_KEY
        openai.organization = organization
        self.model_name = "gpt-4"

    def generate_question(self, data, question_type="DEFAULT"):
        system_message = setup_question_generation(question_type)
        user_message = {
            "role": "user",
            "content": str(data)
        }
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                system_message, # prompt template
                user_message # data
            ],
            temperature=0,
            max_tokens=1024
        )
        return response

    def generate_conversation(self, data, question, conversation_type):
        system_message = setup_conversation(conversation_type)
        user_message = {
            "role": "user",
            "content": str(data) + ". The question is as follows \n" + str(question)
        }
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                system_message,
                user_message
            ],
            temperature=0,
            max_tokens=1024
        )
        return response
