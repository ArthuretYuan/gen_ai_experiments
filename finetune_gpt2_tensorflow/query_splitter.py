import codecs


def extract_question_query(path_file):
    df = codecs.open(
        path_file, 'r').readlines()
    chunks = [df[x:x+4] for x in range(0, len(df), 4)]
    question = []
    query = []
    max_question = 0
    max_anwser = 0
    for f in chunks:
        question.append(f[0])
        question.append(f[1])
        if (len(f[1].split(' ')) > max_question):
            max_question = len(f[1].split(' '))
        query.append(f[2])
        query.append(f[3])
        if (len(f[3].split(' ')) > max_anwser):
            max_anwser = len(f[3].split(' '))
    # -------------- claculate max for max_lenght prameters
    print("anwser ", max_anwser, " question: ", max_question)
    return question, query


def savefile(data_to_save, filepath):
    file = codecs.open(
        filepath, 'w')
    file.write(data_to_save)
    file.close()


def split_data(path_file, save_path, save_file_name_question, save_file_name_query):
    questions, querys = extract_question_query(path_file)
    savefile(''.join(questions), f'{save_path}/{save_file_name_question}')
    savefile(''.join(querys), f'{save_path}/{save_file_name_query}')

# if __name__ == '__main__':
#     path_file = ""
#     save_path = ""
#     save_file_name_question = "data_for_question_model.txt"
#     save_file_name_query = "data_for_query_model.txt"
#     questions, querys = extract_question_query(path_file)
#     savefile(''.join(questions), f'{save_path}/{save_file_name_question}')
#     savefile(''.join(querys), f'{save_path}/{save_file_name_query}')
