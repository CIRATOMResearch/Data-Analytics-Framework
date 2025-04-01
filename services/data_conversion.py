import polars as pl
import os
import json


def convert_to_csv(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    csv_file_path = os.path.splitext(file_path)[0] + '.csv'
    
    try:
        if file_extension == '.xlsx':
            df = pl.read_excel(file_path)
            df.to_csv(csv_file_path, index=False)

            print(f'Конвертировано {file_path} в {csv_file_path}')

        elif file_extension == '.json':
            with open(file_path) as json_file:
                data = json.load(json_file)
                if isinstance(data, dict):
                    data = [data]
                df = pl.DataFrame(data)
                df.to_csv(csv_file_path, index=False)
                print(f'Конвертировано {file_path} в {csv_file_path}')
        else:
            print(f'Неподдерживаемый формат файла: {file_extension}')

    except Exception as e:
        print(f'Произошла ошибка при конвертации: {e}')