# doctor.py

from flask import Blueprint, jsonify

import mysql.connector
doctor = Blueprint('doctor', __name__)

def get_db_connection():

    # 如果连接不存在，则创建一个新的连接

    db_connection = mysql.connector.connect(
            host="116.205.143.194",
            user="root",
            password="zzwzzw",
            database="medical"
    )
    return db_connection

@doctor.route('/getPatients', methods=['GET'])
def getPatients():
    connection=get_db_connection();
    cursor = connection.cursor(dictionary=True)  # 设置为字典模式，使查询结果包含列名

    # 编写 SQL 查询语句
    sql_query = "SELECT * FROM patient"

    # 执行查询
    cursor.execute(sql_query)

    # 获取查询结果
    result = cursor.fetchall()

    # 关闭游标和连接
    cursor.close()
    connection.close()

    # 将结果转换为 JSON 格式并返回
    return jsonify(result)

