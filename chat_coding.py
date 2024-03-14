import os
import streamlit as st

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage

openai_api_key = os.getenv('OPENAI_API_KEY')

MODEL = 'gpt-3.5-turbo'

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

want_to = """너는 아래 내용을 기반으로 질의응답을 하는 로봇이야.
content
{}
"""

content='''
1990년부터 2021년까지의 FIFA 월드컵 대회 정보를 요약하여 제공해 드리겠습니다.

1. **1990년 월드컵 (Italy)**:
    - 개최 날짜: 1990년 6월 8일부터 7월 8일까지
    - 개최 국가: 이탈리아
    - 우승한 나라: 독일 (서독)

2. **1994년 월드컵 (United States)**:
    - 개최 날짜: 1994년 6월 17일부터 7월 17일까지
    - 개최 국가: 미국
    - 우승한 나라: 브라질

3. **1998년 월드컵 (France)**:
    - 개최 날짜: 1998년 6월 10일부터 7월 12일까지
    - 개최 국가: 프랑스
    - 우승한 나라: 프랑스

4. **2002년 월드컵 (South Korea, Japan)**:
    - 개최 날짜: 2002년 5월 31일부터 6월 30일까지
    - 개최 국가: 한국, 일본
    - 우승한 나라: 브라질

5. **2006년 월드컵 (Germany)**:
    - 개최 날짜: 2006년 6월 9일부터 7월 9일까지
    - 개최 국가: 독일
    - 우승한 나라: 이탈리아

6. **2010년 월드컵 (South Africa)**:
    - 개최 날짜: 2010년 6월 11일부터 7월 11일까지
    - 개최 국가: 남아프리카 공화국
    - 우승한 나라: 스페인

7. **2014년 월드컵 (Brazil)**:
    - 개최 날짜: 2014년 6월 12일부터 7월 13일까지
    - 개최 국가: 브라질
    - 우승한 나라: 독일

8. **2018년 월드컵 (Russia)**:
    - 개최 날짜: 2018년 6월 14일부터 7월 15일까지
    - 개최 국가: 러시아
    - 우승한 나라: 프랑스
'''

st.header("월드컵 정보 요약 Q&A")
st.info("1990 - 2018 까지의 월드컵과 관련된 내용을 알아볼 수 있는 Q&A 로봇입니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 백엔드 스쿨 Q&A 로봇입니다. 어떤 내용이 궁금하신가요?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler], model_name=MODEL)
        response = llm([ ChatMessage(role="system", content=want_to.format(content))]+st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))