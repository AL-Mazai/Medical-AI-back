# doctor.py
import math

from flask import Blueprint, jsonify
from flask import request  # 添加这一行来引入 request 模块
import mysql.connector
doctor = Blueprint('doctor', __name__)




#获取数据库连接
def get_db_connection():

    # 如果连接不存在，则创建一个新的连接

    db_connection = mysql.connector.connect(
            host="116.205.143.194",
            user="root",
            password="zzwzzw",
            database="medical"
    )
    return db_connection

#医生登录
@doctor.route('/login',methods=['POST'])
def doctorLogin():
    username = request.form['username']
    password = request.form['password']

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    # 构建 SQL 查询语句
    sql_query = "SELECT * FROM doctor WHERE username=%s and password=%s"
    cursor.execute(sql_query, (username,password))

    # 获取查询结果
    res = cursor.fetchall()
    return jsonify(res)


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

#获取病人列表（可查询）
@doctor.route('/getPatients', methods=['GET'])
def getPatients():
    # 获取搜索关键字，默认为通配符
    keyword = request.args.get('keyword', type=str)
    if keyword== '': keyword= '%'
    # 获取分页参数
    page = request.args.get('page', default=1, type=int)
    page_size = request.args.get('page_size', default=10, type=int)
    doctor_id=request.args.get('doctorId')

    offset = (page - 1) * page_size

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # 构建 SQL 查询语句
    sql_query = """SELECT patient.* FROM patient INNER JOIN doctor_patient 
        ON patient.patient_id = doctor_patient.patient_id 
        WHERE doctor_patient.doctor_id = %s AND patient.username LIKE %s 
        LIMIT %s OFFSET %s"""
    cursor.execute(sql_query, (doctor_id,keyword, page_size, offset))

    # 获取查询结果
    patients = cursor.fetchall()

    # 获取总记录数
    cursor.execute(
        "SELECT COUNT(*) AS total_count FROM patient INNER JOIN doctor_patient ON patient.patient_id = doctor_patient.patient_id WHERE doctor_patient.doctor_id = %s AND patient.username LIKE %s",
        (doctor_id, keyword,))
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


# @doctor.route('/getDiagnosis', methods=['GET'])
# def getDiagnosis():
#     # 获取搜索关键字，默认为通配符
#     # keyword = request.args.get('keyword', type=str)
#     # if keyword== '': keyword= '%'
#     # 获取分页参数
#     page = request.args.get('page', default=1, type=int)
#     page_size = request.args.get('page_size', default=10, type=int)
#     offset = (page - 1) * page_size
#
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
#
#     # 构建 SQL 查询语句
#     sql_query = "SELECT * FROM diagnose_record  LIMIT %s OFFSET %s"
#     cursor.execute(sql_query, (page_size, offset))
#
#     # 获取查询结果
#     patients = cursor.fetchall()
#
#     # 获取总记录数
#     cursor.execute("SELECT COUNT(*) AS total_count FROM patient")
#     total_count = cursor.fetchone()['total_count']
#
#     # 计算总页数
#     total_pages = math.ceil(total_count / page_size)
#
#     # 关闭游标和连接
#     cursor.close()
#     connection.close()
#
#     # 构造响应数据
#     response_data = {
#         'patients': patients,
#         'page': page,
#         'page_size': page_size,
#         'total_pages': total_pages,
#         'total_count': total_count
#     }
#
#     # 返回响应
#     return jsonify(response_data)

@doctor.route('/getDiagnosis', methods=['GET'])
def getDiagnosis():
    # 获取分页参数
    page = request.args.get('page', default=1, type=int)
    page_size = request.args.get('page_size', default=10, type=int)
    doctorId= request.args.get('doctorId', default=10, type=int)
    offset = (page - 1) * page_size

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # 构建 SQL 查询语句，使用 JOIN 连接相关表
    sql_query = """
        SELECT diagnose_record.*, patient.age, patient.gender,
                patient.phone_number, patient.username
        FROM diagnose_record
        JOIN patient ON diagnose_record.patient_id = patient.patient_id
        WHERE doctor_id=%s AND status=1
        LIMIT %s OFFSET %s
    """
    cursor.execute(sql_query, (doctorId,page_size, offset))

    # 获取查询结果
    records = cursor.fetchall()

    # 获取总记录数
    cursor.execute("SELECT COUNT(*) AS total_count FROM diagnose_record")
    total_count = cursor.fetchone()['total_count']

    # 计算总页数
    total_pages = math.ceil(total_count / page_size)

    # 关闭游标和连接
    cursor.close()
    connection.close()

    # 构造响应数据
    response_data = {
        'records': records,
        'page': page,
        'page_size': page_size,
        'total_pages': total_pages,
        'total_count': total_count
    }

    # 返回响应
    return jsonify(response_data)

@doctor.route('/getDiagnosisDetail', methods=['GET'])
def getDiagnosisDetail():
    # 获取诊断参数
    diagnosis_id= request.args.get('diagnosisId', type=int)


    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # 构建 SQL 查询语句，使用 JOIN 连接相关表
    sql_query = """
        SELECT *
        FROM diagnose_record ,patient
        WHERE diagnose_record.patient_id=patient.patient_id and diagnose_record_id=%s and status=1
    """
    cursor.execute(sql_query, (diagnosis_id,))

    # 获取查询结果
    records = cursor.fetchall()

    # 获取总记录数

    # 关闭游标和连接
    cursor.close()
    connection.close()

    # 构造响应数据


    # 返回响应
    return jsonify(records)

@doctor.route('/updateDiagnosisDetail',methods=['PUT'])
def updateDiagnosisDetail():
    diagnosis_id = request.args.get('diagnosisId', type=int)
    diagnose_result=request.args.get('diagnose_result', type=str)
    illness_description=request.args.get('illness_description', type=str)
    treatment_plan=request.args.get('treatment_plan', type=str)

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # 构建 SQL 查询语句，使用 JOIN 连接相关表
    sql_query = """
            UPDATE diagnose_record 
            SET diagnose_result = %s,
            illness_description = %s,
            treatment_plan = %s
            WHERE diagnose_record_id = %s AND status = 1;
        """
    # 执行更新操作
    cursor.execute(sql_query, (diagnose_result,illness_description,treatment_plan,diagnosis_id,))
    # 提交事务
    connection.commit()

    # 检查更新是否成功
    if cursor.rowcount == 0:
        response = {"message": "No record found for the given diagnosis ID or status is not 1"}
    else:
        response = {"message": "Diagnosis record updated successfully"}

    # 关闭游标和连接
    cursor.close()
    connection.close()

    # 返回响应
    return jsonify(response)

@doctor.route('/deleteDiagnosisDetail', methods=['PUT'])
def deleteDiagnosisDetail():
    # 获取诊断参数
    diagnosis_id = request.args.get('diagnosisId', type=int)

    # 获取数据库连接和游标
    connection = get_db_connection()
    cursor = connection.cursor()

    # 构建 SQL 更新语句，将状态标记为 0
    sql_query = """
        UPDATE diagnose_record 
        SET status = 0
        WHERE diagnose_record_id = %s AND status = 1;
    """
    # 执行更新操作
    cursor.execute(sql_query, (diagnosis_id,))
    # 提交事务
    connection.commit()

    # 检查更新是否成功
    if cursor.rowcount == 0:
        response = {"message": "No record found for the given diagnosis ID or status is not 1"}
    else:
        response = {"message": "Diagnosis record marked as deleted successfully"}

    # 关闭游标和连接
    cursor.close()
    connection.close()

    # 返回响应
    return jsonify(response)