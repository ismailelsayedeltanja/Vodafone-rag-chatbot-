import os
import time
import logging
from pathlib import Path
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    #2025-12-7 18:00:22 | INFO | bot started
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/isma3il_chat.log", mode="a", encoding="utf-8"),
          # "w"  يمسح القدي
          # "a"	يضيف على القديم
    ],)
logger = logging.getLogger(__name__)
# logger = logging.getLogger("chatbot")    Logging Config 


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "my-groq-api-key-here")
# os.getenv("VARIABLE_NAME", default_value)
GROQ_MODEL = "llama3-70b-8192" # Meta Llama 3
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

'''
law colab 
from google.colab import userdata
api_key = userdata.get("groq")

'''
'''
########### law h upload file PDF 
!pip install -q langchain langchain-community langchain-huggingface \
langchain-groq chromadb pypdf sentence-transformers


import os
from google.colab import files

os.environ["GROQ_API_KEY"] = "PUT_YOUR_GROQ_API_KEY"
uploaded = files.upload()
pdf_file = list(uploaded.keys())[0]

loader = PyPDFLoader(pdf_file)
docs = loader.load()

print(f"num of pages : {len(docs)}")

'''
# Encoder
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},# vector=1 عشان ال Similarity Search يبقى أدق
)


os.makedirs("data/knowledge_base", exist_ok=True)
os.makedirs("data/vectorstore", exist_ok=True) # مكان vector db
# Directories بيعمل فولدرات 


# Knowledge Base
# انا عملت كنولدج بيز علشان مش لاقي pdf ل فودافون
internet_plans_content = """
باقات انترنت Vodafone

الباقة الاساسية
السعر : 99 جنيه شهريا
الحجم : 10 GB انترنت محمول
الصلاحية : 30 يوم
الشبكة : 4G LTE
مناسبة للاستخدام اليومي الخفيف

الباقة المتوسطة
السعر : 199 جنيه شهريا
الحجم : 30 GB انترنت محمول بالاضافة الى 5 GB للسوشيال ميديا
الصلاحية : 30 يوم
الشبكة : 4G Plus LTE Advanced
مناسبة للاستخدام المتوسط

الباقة المميزة
السعر : 349 جنيه شهريا
الحجم : 80 GB انترنت محمول
الصلاحية : 30 يوم
الشبكة : 5G حيثما توفرت
مناسبة للمستخدمين المكثفين

الباقة غير المحدودة
السعر : 499 جنيه شهريا
الحجم : غير محدود مع سرعة عادلة بعد 100 GB
الصلاحية : 30 يوم
الشبكة : 5G و 4G Plus
مناسبة للمحترفين والعمل عن بعد

طريقة الاشتراك
عبر تطبيق My Vodafone
عبر الكود 888 نجمة
عبر الموقع الرسمي vodafone.com.eg
زيارة اقرب فرع Vodafone
"""

calling_plans_content = """
خطط الاتصال والرسائل Vodafone

خطة Red Basic
السعر : 149 جنيه شهريا
الدقائق : 300 دقيقة لجميع الشبكات
الرسائل : 300 رسالة نصية
انترنت : 5 GB
مناسبة للاستخدام الخفيف

خطة Red Plus
السعر : 299 جنيه شهريا
الدقائق : غير محدودة داخل Vodafone بالاضافة الى 500 دقيقة للشبكات الاخرى
الرسائل : غير محدودة
انترنت : 20 GB
مكالمات واتساب مجانية

خطة Red Max
السعر : 499 جنيه شهريا
الدقائق : غير محدودة لجميع الشبكات
الرسائل : غير محدودة
انترنت : 60 GB
بيانات دولية : 2 GB
تحويل الخط مجاني

خطة Business
السعر : 799 جنيه شهريا
الدقائق : غير محدودة
الانترنت : 100 GB
خدمات اضافية للاعمال
دعم اولوية 24 ساعة طوال الاسبوع
تقارير الاستخدام الشهرية
"""

technical_support_content = """
الدعم الفني Vodafone

مشكلة الانترنت البطيء
الخطوة الاولى : تحقق من قوة اشارة الشبكة يجب ان تكون ثلاثة اعمدة على الاقل
الخطوة الثانية : اعد تشغيل الهاتف وانتظر دقيقة كاملة
الخطوة الثالثة : تحقق من استهلاك الباقة عبر كود 101 نجمة او التطبيق
الخطوة الرابعة : تاكد ان APN مضبوط على vodafone.com.eg
الخطوة الخامسة : اذا استمرت المشكلة اتصل بالدعم على 888

مشكلة الانترنت لا يعمل
الخطوة الاولى : تاكد ان البيانات المحمولة مفعلة في الاعدادات
الخطوة الثانية : تحقق من الباقة المشتركة عبر كود 101 نجمة
الخطوة الثالثة : اعد تشغيل الهاتف
الخطوة الرابعة : تحقق من اعدادات APN
الخطوة الخامسة : اتصل بالدعم الفني على 888

مشكلة الاتصال
الخطوة الاولى : تحقق من وضع الطيران يجب ان يكون مغلقا
الخطوة الثانية : تاكد من ان شريحة SIM مثبتة بشكل صحيح
الخطوة الثالثة : اعد تشغيل الهاتف
الخطوة الرابعة : تحقق من الرصيد عبر كود 888 نجمة
الخطوة الخامسة : في حالة فقد الشبكة تماما اتصل بـ 888

اعدادات APN لـ Vodafone مصر
APN : vodafone.com.eg
Username : vodafone
Password : vodafone
MCC : 602
MNC : 02

طريقة نقل الرقم MNP
احتفظ برقمك عند التحويل لـ Vodafone
قدم بطاقة الهوية الوطنية
تواصل مع اي فرع Vodafone
العملية تستغرق ثلاثة ايام عمل
رسوم النقل مجانية

وسائل التواصل
الدعم الهاتفي : 888 مجاني
واتساب : 01001888888
البريد الالكتروني : cs@vodafone.com.eg
الموقع : vodafone.com.eg
"""

vodafone_pay_content = """
Vodafone Pay خدمة الدفع الرقمي

ما هو Vodafone Pay
Vodafone Pay هو محفظة رقمية تتيح لك ارسال واستقبال الاموال وسداد الفواتير وشراء الباقات وانجاز المعاملات المالية بامان وسهولة من هاتفك

طريقة التسجيل
الخطوة الاولى : تحميل تطبيق Vodafone Pay من App Store او Google Play
الخطوة الثانية : ادخل رقم هاتف Vodafone الخاص بك
الخطوة الثالثة : ادخل رقمك القومي للتحقق
الخطوة الرابعة : انشئ PIN سري من ستة ارقام
الخطوة الخامسة : ابدا الاستخدام فورا

الخدمات المتاحة
تحويل الاموال لاي رقم Vodafone
سداد فواتير الكهرباء والغاز والمياه
شحن محافظ الشبكات الاخرى
الشراء اونلاين في الاف المواقع
سحب النقود من ماكينات ATM
استقبال مرتبات ومدفوعات

الحدود والرسوم
الحد اليومي للتحويل : 10000 جنيه للفئة الاساسية
الحد الشهري : 40000 جنيه
رسوم التحويل : مجاني بين محافظ Vodafone Pay
سحب الاموال : واحد بالمئة من المبلغ بحد ادنى جنيهين
ترقية الحساب توفر حدودا اعلى
"""

# habdaa ahawel el knowledge base le file.txt
with open("data/knowledge_base/internet_plans.txt", "w", encoding="utf-8") as f:
    f.write(internet_plans_content)

with open("data/knowledge_base/calling_plans.txt", "w", encoding="utf-8") as f:
    f.write(calling_plans_content)

with open("data/knowledge_base/technical_support.txt", "w", encoding="utf-8") as f:
    f.write(technical_support_content)

with open("data/knowledge_base/vodafone_pay.txt", "w", encoding="utf-8") as f:
    f.write(vodafone_pay_content)
# encoding="utf-8" مهم للعربي خد بالك

# data pre
def load_documents(data_dir="data/knowledge_base"):
    data_path = Path(data_dir)  # tahwel el txt to Documents 
    documents = []      #collect all documents 

    for txt_file in data_path.glob("*.txt"): # .txt ابحث عن كل الملفات التي تنتهي  
        try:  # تحميل ملف نصي وتحويله إلى Documents.
            loader = TextLoader(str(txt_file), encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e: # لو حصل Error خزنه داخل e
            logger.error(f"error : {txt_file.name} : {e}")

    for pdf_file in data_path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"erooorrrrr : {pdf_file.name} : {e}")

    return documents


# hamel chunk
def split_documents(documents, chunk_size=501, chunk_overlap=60):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "،", " ", "", "-"],
              )
    return splitter.split_documents(documents)


def build_vectorstore(embedding_model, force_rebuild=False):
    vectorstore_path = Path("data/vectorstore")

    if vectorstore_path.exists() and not force_rebuild:
        # لو الفايل موجود + مش عايز أعيد بناءه
        vectorstore = FAISS.load_local( #تحميل index محفوظ على الجهاز
            str(vectorstore_path),
            embedding_model, #  لو غيرت بناء embedding model لازم تعيد بناء vectorstore
            allow_dangerous_deserialization=True, # أنا واثق من الملف ده، فكّه عادي 
                                       )
        return vectorstore
        
    # تحميل الملفات وتقسيمها
    documents = load_documents()
    chunks = split_documents(documents)
    
     # Semantic Search
    vectorstore = FAISS.from_documents(
        documents=chunks, 
        embedding=embedding_model,                              
        distance_strategy=DistanceStrategy.COSINE)
                         
    
  # حفظه على الجها
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_path)) # علشان مش كل شويه اعمل شانك وانبدنج 

    return vectorstore


vectorstore = build_vectorstore(embedding_model)


def retrieve_context(query, vectorstore, top_k=4, score_threshold=0.3):
    # search باستخدام cosine similarity
    # يبحث داخل vector database
    # السؤال يتحول Vector
    results = vectorstore.similarity_search_with_relevance_scores(
        query=query, #الفرق بينها وبين similarity_search() ترجع (doc, score)
        k=top_k)

    filtered_docs = []
    filtered_scores = []
    # يحسب similarity
    # يجلب أفضل chunks
    for doc, similarity in results:
        
        # يفلتر النتائج الضعيفة
        if similarity >= score_threshold:  # هل النتيجة دي قوية كفاية
            filtered_docs.append(doc)
            filtered_scores.append(similarity)
  
    return filtered_docs, filtered_scores




# Decooooder 
#  لازم نحول filtered_docs , filtered_scores LLM نص مرتب قبل
 # Prompt Engineering
 # retrieved chunks
# يقرأ metadata
# ينسق الـ scores
 # يبني prompt context مرتب
# يحول الـ documents إلى نص جاهز للـ LLM
def format_context(documents, scores):
    if not documents:  # if len(documents) == 0:
        return "No relevant context found"  # fallback messages لازم تكون meaningful

    context_parts = []
    for i, (doc, score) in enumerate(zip(documents, scores), 1):
        source = doc.metadata.get("source", "un_known")
        source_name = Path(source).name if source != "unknown" else "unknown"
        context_parts.append(f"[from {i}: {source_name} |  : {score:.2%}]\n{doc.page_content.strip()}")

    return "\n---\n".join(context_parts)

'''
respose
[مصدر 1: internet.txt | 91.22%]
الباقة المميزة
80 GB
'''


#هنا  Client للتواصل مع Groq Api

# يبني الـ prompt
# يضيف conversation memory
# يضيف retrieved context
# يرسل الطلب لـ Groq LLM
# يولد الرد النهائي
# يتحكم في الـ tone والـ hallucination
# يمثل الجزء التوليدي من الـ RAG

groq_client = Groq(api_key=GROQ_API_KEY) # يبعت requests
                                         # يستقبل responses
def generate_response(query, context, conversation_history):
    system_prompt = # توليد الرد النهائي للمستخدم
    """
    ممنوع تجاوب من دماغك
    انت مساعد خدمة عملاء Vodafone المتخصص
    مهمتك مساعدة العملاء في باقات الانترنت وخطط الاتصال وVodafone Pay والدعم الفني.
    استخدم المعلومات المقدمة فقط اذا لم تجد اجابة قل ذلك واقترح الاتصال على 888
    كن وديا ومحترفا واستخدم اللغة العربية البسيطة
    """

    messages = []
    for msg in conversation_history[-10:]: #آخر 10 رسائل  
        messages.append({"role": msg["role"], "content": msg["content"]})
   # هنا بعمل Chat Format
    messages.append({
        "role": "user",
        "content": f"from chunk num :\n{context}\n\nسؤال العميل:\n{query}\n\naجب بناء على المعلومات المقدمة فقط."
    })

    try:  #هنا يتم دمج:system prompt مع conversation messages
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=900,
              temperature=0.2,
        )
        return response.choices[0].message.content # أول رد
    except Exception as e:
        logger.error(f"eroor  Groq : {e}") #لو حصل خطأ خزنه داخل e

        return f"eroor : {str(e)}"



conversation_history = []  # دي Memory بسيطة للمحادثة
def chat(user_message, top_k=5):
    global conversation_history

    start_time = time.time() #Response Latency
    # هنا بدأ  Semantic Search
    relevant_docs, scores = retrieve_context(user_message, vectorstore, top_k)
    context = format_context(relevant_docs, scores)
    response_text = generate_response(user_message, context, conversation_history)
     
    #Conversation Buffer Memory
    #تحفظ رسالة المستخدم
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": response_text})
    conversation_history = conversation_history[-20:] #Conversation Memory Limit
    # يتم إنشاء Dictionary لكل مصدر
    sources = []
    for doc, score in zip(relevant_docs, scores):
        sources.append({
            "file": Path(doc.metadata.get("source", "")).name,
            "relevance": f"{score:.2%}", # similarity نسبة مئوية
            "preview": doc.page_content[:150], # ول 150 حرف 
        })

    # هنا ترجع النتيجة النهائية بشكل Structured
    return {
        "answer": response_text,
        "sources": sources,
        "response_time": f"{round(time.time() - start_time, 2)} sec",#مدة تنفيذ الطلب
        "docs_retrieved": len(relevant_docs),    #عدد الـ chunks المستخدمة
                                                  #تقريب لـ رقمين عشريين
           }


def clear_history():
    global conversation_history
    conversation_history = []
    logger.info("Conversation history cleared")
    #print(" deleted done ya 3am mafesh ")



test_questions = [
    "ما هي باقات الانترنت المتاحة واسعارها؟",
    "الانترنت عندي بطيء كيف احل المشكلة؟",
    "كيف اشترك في Vodafone Pay؟",
    "ما هو رقم خدمة العملاء؟",
]

for question in test_questions:
    print(f"\nالسؤال : {question}")
    print("-" * 40)
    result = chat(question)
    print(f"الرد :\n{result['answer']}")
    print(f"وقت الاستجابة : {result['response_time']}")
    print(f"مستندات مسترجعة : {result['docs_retrieved']}")
    if result["sources"]:
        print("المصادر :")
        for src in result["sources"]:
            print(f"  - {src['file']} | الصلة : {src['relevance']}")
    print("=" * 60)


def run_interactive_chat():
    print("\n" + "=" * 60)
    print("مساعد خدمة عملاء Vodafone")
    print("اكتب سؤالك او اكتب خروج للانهاء")
    print("=" * 60 + "\n")

    while True:
        user_input = input("انت : ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["خروج", "quit", "exit"]:
            print("شكرا لاستخدامك خدمة Vodafone. مع السلامة.")
            break

        if user_input.lower() in ["مسح", "clear"]:
            clear_history()
            continue

        print("\nloading  ...")
        result = chat(user_input)
        print(f"\nhelper_ :\n{result['answer']}")
        print(f"\ntime_  : {result['response_time']}")
        print("-" * 60 + "\n")


# run_interactive_chat()


def gradio_chat(user_message, history):
    if not user_message.strip():
        return "", history

    result = chat(user_message)
    response = result["answer"]

    if result["sources"]:
        response += "\n\n---\nالمصادر :\n"
        for i, src in enumerate(result["sources"], 1):
            response += f"{i}. {src['file']} | الصلة : {src['relevance']}\n"

    response += f"\nوقت الاستجابة : {result['response_time']}"
    history.append((user_message, response))
    return "", history


def gradio_clear():
    clear_history()
    return [], []


with gr.Blocks(title="Vodafone Customer Service Chatbot") as demo:

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_component = gr.Chatbot(label="المحادثة", height=450, rtl=True)

            with gr.Row():
                msg_input = gr.Textbox(
                    label="اكتب سؤالك",
                    placeholder="مثال : ما هي باقات الانترنت المتاحة؟",
                    scale=4,
                    rtl=True,
                )
                send_btn = gr.Button("ارسال", variant="primary", scale=1)

            clear_btn = gr.Button("مسح المحادثة", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### اسئلة مقترحة")
            suggested = [
                "ما هي باقات الانترنت واسعارها؟",
                "كيف اشترك في Red Plus؟",
                "الانترنت بطيء كيف احل المشكلة؟",
                "ما هي اعدادات APN؟",
                "كيف اسجل في Vodafone Pay؟",
                "ما هو رقم خدمة العملاء؟",
                "كيف انقل رقمي الى Vodafone؟",
            ]
            for question in suggested:
                btn = gr.Button(question, size="sm")
                btn.click(fn=lambda q=question: q, outputs=msg_input)

    send_btn.click(fn=gradio_chat, inputs=[msg_input, chatbot_component], outputs=[msg_input, chatbot_component])
    msg_input.submit(fn=gradio_chat, inputs=[msg_input, chatbot_component], outputs=[msg_input, chatbot_component])
    clear_btn.click(fn=gradio_clear, outputs=[chatbot_component, chatbot_component])

# demo.launch(server_name="0.0.0.0", server_port=7860)

print("تم تعريف واجهة Gradio")


app = FastAPI(
    title="Vodafone RAG Chatbot API",
    description="Vodafone Customer Service Chatbot - اسماعيل السيد",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")


class ChatResponse(BaseModel):
    answer: str
    sources: list
    response_time: str
    docs_retrieved: int
    session_id: str


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Vodafone RAG Chatbot API",
        "developer": "som3aa ",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vectorstore": "loaded" if vectorstore else "not loaded",
        "groq_model": GROQ_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = chat(request.message)
        result["session_id"] = request.session_id
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"خطأ : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/history")
async def clear_chat_history():
    clear_history()
    return {"message": "was dweleted   ", "status": "success"}


@app.get("/stats")
async def get_stats():
    return {
        "conversation_turns": len(conversation_history) // 2,
        "vectorstore_loaded": vectorstore is not None,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "groq_model": GROQ_MODEL,
    }


@app.get("/search")
async def search_documents(query: str, top_k: int = 5):
    docs, scores = retrieve_context(query, vectorstore, top_k)
    results = []
    for doc, score in zip(docs, scores):
        results.append({
            "content": doc.page_content[:500],
            "source": doc.metadata.get("source", ""),
            "relevance_score": round(score, 4),
        })
    return {"query": query, "results": results, "total_found": len(results)}


# uvicorn.run(app, host="0.0.0.0", port=8000)

print("FastAPI done")
print(" جاهز -  :   yalla penaaaaaaaaaa")
