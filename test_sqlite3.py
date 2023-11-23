import sqlite3
import os
# dir = os.path.abspath(os.curdir)
sqliteConnection = sqlite3.connect(r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\InnopolysFinal-main\SQLite_database.db')
cursor = sqliteConnection.cursor()

cursor.execute('SELECT * FROM table_1')
data = cursor.fetchall()

for row in data:
    print(row)









# sqlite_insert_blob_query = """ INSERT INTO table_1                            
# (input_image, output_image, model_prediction, datetime) VALUES (?, ?, ?, ?)"""
            

# data_tuple = (1, 132323, 'stringParameter', '1111')
# cursor.execute(sqlite_insert_blob_query, data_tuple)
sqliteConnection.commit()