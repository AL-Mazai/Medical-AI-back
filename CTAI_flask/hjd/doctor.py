# doctor.py
import math

from flask import Blueprint, jsonify
from flask import request  # 添加这一行来引入 request 模块
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

# @doctor.route('/getPatients', methods=['GET'])
# def getPatients():
#     connection=get_db_connection();
#     cursor = connection.cursor(dictionary=True)  # 设置为字典模式，使查询结果包含列名
#
#     # 编写 SQL 查询语句
#     sql_query = "SELECT * FROM patient"
#
#     # 执行查询
#     cursor.execute(sql_query)
#
#     # 获取查询结果
#     result = cursor.fetchall()
#
#     # 关闭游标和连接
#     cursor.close()
#     connection.close()
#
#     # 将结果转换为 JSON 格式并返回
#     return jsonify(result)

@doctor.route('/getPatients', methods=['GET'])
def getPatients():
    # 获取搜索关键字，默认为通配符
    keyword = request.args.get('keyword', type=str)
    if keyword== '': keyword= '%'
    # 获取分页参数
    page = request.args.get('page', default=1, type=int)
    page_size = request.args.get('page_size', default=10, type=int)
    offset = (page - 1) * page_size

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # 构建 SQL 查询语句
    sql_query = "SELECT * FROM patient WHERE username LIKE %s LIMIT %s OFFSET %s"
    cursor.execute(sql_query, (keyword, page_size, offset))

    # 获取查询结果
    patients = cursor.fetchall()

    # 获取总记录数
    cursor.execute("SELECT COUNT(*) AS total_count FROM patient WHERE username LIKE %s", (keyword,))
    total_count = cursor.fetchone()['total_count']

    # 计算总页数
    total_pages = math.ceil(total_count / page_size)

    # 关闭游标和连接
    cursor.close()
    connection.close()

    # 构造响应数据
    response_data = {
        'patients': patients,
        'page': page,
        'page_size': page_size,
        'total_pages': total_pages,
        'total_count': total_count
    }

    # 返回响应
    return jsonify(response_data)

