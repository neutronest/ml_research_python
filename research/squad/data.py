import json
from collections import namedtuple

Answer = namedtuple("Answer", ["text", "answer_start"])
QA = namedtuple("QA", ["question", "id", "answers", "is_impossible"])
Paragragh = namedtuple("Paragraph", ["qas", "context"])
SquadData = namedtuple("SquadData", ["title", "paragraphs"])


def main():
    with open("./data/squadv2.0/train-v2.0.json") as file_ob:
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


if __name__ == "__main__":
    main()