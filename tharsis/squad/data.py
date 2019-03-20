import json
from collections import namedtuple

Answer = namedtuple("Answer", ["text", "answer_start"])
QA = namedtuple("QA", ["question", "id", "answers", "is_impossible"])
Paragragh = namedtuple("Paragraph", ["qas", "context"])
SquadData = namedtuple("SquadData", ["title", "paragraphs"])

def generate_squad_data_list(json_file_path):
    with open(json_file_path) as file_ob:
        data = json.load(file_ob)
    
    squad_data_list = []
    for data_item in data.get("data"):
        paragraphs = []
        for paragraph_data in data_item.get("paragraphs"):
            qa_data_list = paragraph_data.get("qas")

            qas = []
            for qa_data in qa_data_list:
                answer_data_list = qa_data.get("answers")
                answer_list = []
                for answer_data in answer_data_list:
                    answer_list.append(
                        Answer(answer_data.get("text"), answer_data.get("answer_start"))
                    )
                qa = QA(
                    qa_data.get("question"),
                    qa_data.get("id"),
                    answer_list,
                    qa_data.get("is_impossible")
                )
                qas.append(qa)
            paragraph = Paragragh(qas, paragraph_data.get("context"))
            paragraphs.append(paragraph)
        squad_data = SquadData(title=data_item["title"], paragraphs=paragraphs)
        squad_data_list.append(squad_data)
    return squad_data_list

def generate_tokens_map_by_squad_data_list(squad_data_list):
    token_map = {}
    for squad_data in squad_data_list:
        for paragraph in squad_data.paragraphs:
            context = paragraph.context
            question_answers = paragraph.qas

            context_tokens = context.split(" ")
            for token in context_tokens:
                if token_map.get(token) is None:
                    token_map[token] = 1
            for qa in question_answers:
                if len(qa.answers) <= 0:
                    continue
                question_text = qa.question
                answer_text = qa.answers[0].text
                for token in question_text.split(" "):
                    if token_map.get(token) is None:
                        token_map[token] = 1
                for token in answer_text.split(" "):
                    if token_map.get(token) is None:
                        token_map[token] = 1
    return token_map

def generate_tokens_map_by_flatten_pqa_data(flatten_data_list):
    """
    flatten_data_list: List[(paragraph, question, answer)]
    """
    token_map = {}
    for (paragraph_text, question_text, answer_text) in flatten_data_list:
        paragraph_tokens = paragraph_text.split(" ")
        question_tokens = question_text.split(" ")
        answer_tokens = answer_text.split(" ")
        for token  in paragraph_tokens + question_tokens + answer_tokens:
            if token_map.get(token) is None:
                token_map[token] = 1
    return token_map




def generate_paragraph_question_answer_data(squad_data_list):
    
    paragraph_question_answer_data_list = []
    for squad_data in squad_data_list:
        for paragraph in squad_data.paragraphs:
            context = paragraph.context
            question_answers = paragraph.qas
            for qa in question_answers:
                if len(qa.answers) <= 0:
                    continue
                question_text = qa.question
                answer_text = qa.answers[0].text
                paragraph_question_answer_data_list.append(
                    (context, question_text, answer_text)

                )
    return paragraph_question_answer_data_list


def main():
    squad_data_list = generate_squad_data_list("./data/squadv2.0/train-v2.0.json")
    data = generate_paragraph_question_answer_data(squad_data_list)
    return

if __name__ == "__main__":
    main()