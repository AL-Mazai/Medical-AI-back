# doctor.py
from zhipuai import ZhipuAI
from flask import Blueprint, jsonify
from flask import request  # 添加这一行来引入 request 模块

AI = Blueprint('AI', __name__)


@AI.route('/AIGenerate',methods=['POST'])
def AIGenerate():
    diagnose_result=request.form.get('diagnose_result', type=str)
    illness_description=request.form.get('illness_description', type=str)
    treatment_plan=request.form.get('treatment_plan', type=str)
    client = ZhipuAI(api_key="cb2e7c320a1cd6e7d3936f5c9ef73eff.69VRzkYzQAmbZ4ez")  # 请填写您自己的APIKey

    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": "请用我以下资料生成医疗文档 我的资料其中包含疾病名 患者描述 医生建议 "},
            {"role": "user", "content": "疾病诊断名为:"+diagnose_result+"诊断名为:"+illness_description+"医生建议为:"+treatment_plan},
            {"role": "user",
             "content": ",直接根据上述内容给出描述即可 不要有不确定的内容 我是直接要放到网页上的 直接写出内容即可 不要有特殊符号 不要有类似：‘以下是我给出的诊断文档’这种语言 直接给出回答 "},
        ],
    )
    answer = response.choices[0].message.content
    print(answer)
    return jsonify(answer)

@AI.route('/AITalk',methods=['POST'])
def AITalk():
    user_input = request.form.get('user_input', type=str)
    client = ZhipuAI(api_key="cb2e7c320a1cd6e7d3936f5c9ef73eff.69VRzkYzQAmbZ4ez")  # 填写您自己的APIKey
    conversation = []  # 用于保存对话上下文的列表
    conversation.append({"role": "user", "content": user_input})  # 将用户输入添加到对话上下文中
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=conversation,  # 使用包含对话上下文的消息列表
    )
    answer = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": answer})  # 将助手的回复添加到对话上下文中
    return jsonify(answer)
# if __name__ == '__main__':
#     # client = ZhipuAI(api_key="596c1d3621e41f40477264b097ed6e22.Q4q3ijf7Ak5dmGiJ")  # 填写您自己的APIKey
#     # while True:
#     #     prompt = input("user:")
#     #     response = client.chat.completions.create(
#     #         model="glm-4",  # 填写需要调用的模型名称
#     #         messages=[
#     #             {"role": "user", "content": prompt}
#     #         ],
#     #     )
#     #     answer = response.choices[0].message.content
#     #     print("ZhipuAI:", answer)
#     client = ZhipuAI(api_key="596c1d3621e41f40477264b097ed6e22.Q4q3ijf7Ak5dmGiJ")  # 请填写您自己的APIKey
#     conversation = []  # 用于保存对话上下文的列表
#     while True:
#         prompt = input("user:")
#         conversation.append({"role": "user", "content": prompt})  # 将用户输入添加到对话上下文中
#         response = client.chat.completions.create(
#             model="glm-4",  # 填写需要调用的模型名称
#             messages=conversation,  # 使用包含对话上下文的消息列表
#         )
#         answer = response.choices[0].message.content
#         print("ZhipuAI:", answer)
#         conversation.append({"role": "assistant", "content": answer})  # 将助手的回复添加到对话上下文中