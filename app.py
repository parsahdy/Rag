import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from pdf_processor import process_pdfs, get_pdf_text
from rag_manager import RAGManager, setup_embeddings, get_llm
from study_planner import create_study_plan


st.set_page_config(
    page_title="سیستم مشاور هوشمند",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @font-face {
        font-family: 'Vazir';
        src: url('https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/Vazir.woff2') format('woff2');
    }
    * {
        font-family: 'Vazir', sans-serif !important;
    }
    .main {
        direction: rtl;
        text-align: right;
    }
    .stButton button {
        width: 100%;
    }
    .css-1kyxreq {
        justify-content: flex-end;
    }
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = None
if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False
if "data_dir" not in st.session_state:
    st.session_state.data_dir = "data"
    os.makedirs(st.session_state.data_dir, exist_ok=True)
if "db_dir" not in st.session_state:
    st.session_state.db_dir = "vectordb"
    os.makedirs(st.session_state.db_dir, exist_ok=True)


with st.sidebar:
    st.title("سیستم مشاور هوشمند")
    st.write("این سیستم به شما کمک می‌کند تا با استفاده از هوش مصنوعی به سؤالات خود پاسخ دهید و برنامه مطالعاتی بسازید.")
    
    
    tab_option = st.radio(
        "بخش مورد نظر را انتخاب کنید:",
        ["مشاوره و گفتگو", "برنامه هفتگی مطالعه"]
    )
    
    
    st.subheader("آپلود فایل‌های PDF")
    uploaded_files = st.file_uploader("فایل‌های PDF خود را بارگذاری کنید", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and not st.session_state.pdfs_processed and st.button("پردازش فایل‌ها"):
        with st.spinner("در حال پردازش فایل‌های PDF..."):
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(st.session_state.data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            
            process_pdfs(st.session_state.data_dir, st.session_state.db_dir)
            
            
            embeddings = setup_embeddings()
            llm = get_llm()
            st.session_state.rag_manager = RAGManager(st.session_state.db_dir, llm, embeddings)
            
            st.session_state.pdfs_processed = True
            st.success("فایل‌ها با موفقیت پردازش شدند!")
            st.rerun()
    
    
    if st.session_state.pdfs_processed and st.button("شروع مجدد گفتگو"):
        st.session_state.messages = []
        st.rerun()


st.title("سیستم مشاور هوشمند مبتنی بر RAG")


if tab_option == "مشاوره و گفتگو":
    st.header("گفتگو با مشاور هوشمند")
    
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    
    if prompt := st.chat_input("سؤال خود را بپرسید"):
        if not st.session_state.pdfs_processed:
            st.error("لطفاً ابتدا فایل‌های PDF را بارگذاری و پردازش کنید.")
            st.stop()
        
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        
        with st.chat_message("assistant"):
            with st.spinner("در حال تهیه پاسخ..."):
                response = st.session_state.rag_manager.get_response(prompt)
                st.markdown(response)
        
        
        st.session_state.messages.append({"role": "assistant", "content": response})


elif tab_option == "برنامه هفتگی مطالعه":
    st.header("ساخت برنامه هفتگی مطالعه")
    
    
    with st.form("study_plan_form"):
        st.write("لطفاً اطلاعات زیر را برای ساخت برنامه هفتگی وارد کنید:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            student_name = st.text_input("نام دانش‌آموز:")
            grade = st.selectbox("پایه تحصیلی:", ["هفتم", "هشتم", "نهم", "دهم", "یازدهم", "دوازدهم"])
            field = st.selectbox("رشته تحصیلی:", ["عمومی", "ریاضی", "تجربی", "انسانی", "فنی و حرفه‌ای"])
        
        with col2:
            start_date = st.date_input("تاریخ شروع برنامه:", datetime.now())
            daily_hours = st.slider("ساعات مطالعه روزانه:", 1, 12, 4)
            goal = st.text_input("هدف اصلی (مثال: کنکور، امتحان نهایی، و غیره):")
        
        
        st.subheader("دروس مورد نظر:")
        col1, col2, col3 = st.columns(3)
        
        subjects = []
        priorities = []
        
        with col1:
            subject1 = st.text_input("درس 1:")
            subject2 = st.text_input("درس 2:")
            subject3 = st.text_input("درس 3:")
            if subject1: 
                subjects.append(subject1)
                priorities.append(st.slider(f"اولویت {subject1}:", 1, 10, 5, key="p1"))
            if subject2: 
                subjects.append(subject2)
                priorities.append(st.slider(f"اولویت {subject2}:", 1, 10, 5, key="p2"))
            if subject3: 
                subjects.append(subject3)
                priorities.append(st.slider(f"اولویت {subject3}:", 1, 10, 5, key="p3"))
        
        with col2:
            subject4 = st.text_input("درس 4:")
            subject5 = st.text_input("درس 5:")
            subject6 = st.text_input("درس 6:")
            if subject4: 
                subjects.append(subject4)
                priorities.append(st.slider(f"اولویت {subject4}:", 1, 10, 5, key="p4"))
            if subject5: 
                subjects.append(subject5)
                priorities.append(st.slider(f"اولویت {subject5}:", 1, 10, 5, key="p5"))
            if subject6: 
                subjects.append(subject6)
                priorities.append(st.slider(f"اولویت {subject6}:", 1, 10, 5, key="p6"))
        
        with col3:
            subject7 = st.text_input("درس 7:")
            subject8 = st.text_input("درس 8:")
            subject9 = st.text_input("درس 9:")
            if subject7: 
                subjects.append(subject7)
                priorities.append(st.slider(f"اولویت {subject7}:", 1, 10, 5, key="p7"))
            if subject8: 
                subjects.append(subject8)
                priorities.append(st.slider(f"اولویت {subject8}:", 1, 10, 5, key="p8"))
            if subject9: 
                subjects.append(subject9)
                priorities.append(st.slider(f"اولویت {subject9}:", 1, 10, 5, key="p9"))
        
        
        notes = st.text_area("یادداشت‌ها و محدودیت‌های زمانی (مثال: کلاس‌های فوق‌برنامه، زمان‌های غیرقابل مطالعه):")
        
        
        submitted = st.form_submit_button("ساخت برنامه هفتگی")
    
    
    if submitted:
        if not subjects:
            st.error("لطفاً حداقل یک درس وارد کنید.")
        else:
            with st.spinner("در حال ساخت برنامه مطالعاتی..."):
                student_info = {
                    "name": student_name,
                    "grade": grade,
                    "field": field,
                    "goal": goal,
                    "daily_hours": daily_hours,
                    "start_date": start_date,
                    "subjects": subjects,
                    "priorities": priorities,
                    "notes": notes
                }
                
                
                if st.session_state.pdfs_processed:
                    study_plan_data = create_study_plan(student_info, st.session_state.rag_manager)
                else:
                    llm = get_llm()
                    study_plan_data = create_study_plan(student_info, None, llm)
                
                
                st.success("برنامه مطالعاتی با موفقیت ایجاد شد!")
                
                
                st.subheader("اطلاعات دانش‌آموز")
                st.write(f"**نام:** {student_name}")
                st.write(f"**پایه تحصیلی:** {grade}")
                st.write(f"**رشته:** {field}")
                st.write(f"**هدف:** {goal}")
                
                
                st.subheader("برنامه هفتگی مطالعه")
                
                
                df_plan = pd.DataFrame(study_plan_data)
                st.dataframe(df_plan, use_container_width=True)
                
                
                csv = df_plan.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="دانلود برنامه (CSV)",
                    data=csv,
                    file_name=f"برنامه_مطالعاتی_{student_name}_{start_date}.csv",
                    mime="text/csv",
                )
                
                
                st.subheader("توصیه‌های مطالعاتی")
                st.markdown("""
                1. سعی کنید هر روز در زمان مشخصی مطالعه کنید تا عادت منظمی ایجاد شود.
                2. بین جلسات مطالعه، استراحت کوتاه (5-15 دقیقه) داشته باشید.
                3. محیط مطالعه آرام و بدون عامل حواس‌پرتی انتخاب کنید.
                4. قبل از شروع مطالعه، گوشی و سایر عوامل حواس‌پرتی را دور کنید.
                5. در پایان هر جلسه مطالعه، خلاصه‌ای از آنچه یاد گرفته‌اید بنویسید.
                """)