
import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fpdf import FPDF


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Customer Review Intelligence",
    page_icon="🤖",
    layout="wide"
)


# =========================================
# GEMINI CONFIGURATION
# =========================================

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# =========================================
# TITLE SECTION
# =========================================

st.title("🤖 AI Customer Review Intelligence System")

st.info("""
🚀 AI-powered customer review intelligence platform for businesses.

Upload review datasets and instantly generate:
- Sentiment insights
- Customer complaints
- Business recommendations
- AI-generated analytics reports
""")


st.markdown("""
### Transform Customer Reviews into Business Intelligence

Upload customer reviews and get:
- 📊 Sentiment Analytics
- 🤖 AI Business Insights
- ⚠️ Top Customer Complaints
- 💡 Strategic Recommendations
""")


# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("📌 Navigation")

st.sidebar.info("""
AI-Powered Review Analytics Platform

Features:
- Sentiment Analysis
- AI Insights
- Complaint Detection
- Business Recommendations
""")


# =========================================
# FILE UPLOAD
# =========================================

st.divider()

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


# =========================================
# MAIN PROCESSING
# =========================================

if uploaded_file is not None:

    # =====================================
    # READ DATA
    # =====================================

    df = pd.read_csv(uploaded_file)

    st.success("✅ CSV File Uploaded Successfully!")

    st.divider()


    # =====================================
    # DATASET SUMMARY
    # =====================================

    st.subheader("📂 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", len(df))
    col2.metric("Total Products", df['ProductId'].nunique())
    col3.metric("Total Users", df['UserId'].nunique())


    # =====================================
    # DATA PREVIEW
    # =====================================

    st.subheader("🗂️ Dataset Preview")

    st.dataframe(df.head())


    # =====================================
    # REVIEW SCORE DISTRIBUTION
    # =====================================

    sentiment_counts = df['Score'].value_counts()

    st.divider()

    st.subheader("📊 Review Score Distribution")

    st.bar_chart(sentiment_counts)


    # =====================================
    # SENTIMENT CLASSIFICATION
    # =====================================

    def classify_sentiment(score):

        if score >= 4:
            return "Positive"

        elif score == 3:
            return "Neutral"

        else:
            return "Negative"


    df['Sentiment'] = df['Score'].apply(classify_sentiment)


    # =====================================
    # SENTIMENT COUNTS
    # =====================================

    positive_count = (df['Sentiment'] == 'Positive').sum()

    negative_count = (df['Sentiment'] == 'Negative').sum()

    neutral_count = (df['Sentiment'] == 'Neutral').sum()

    total_reviews = len(df)

    positive_percent = round((positive_count / total_reviews) * 100, 2)

    negative_percent = round((negative_count / total_reviews) * 100, 2)

    neutral_percent = round((neutral_count / total_reviews) * 100, 2)


    # =====================================
    # METRICS
    # =====================================

    st.divider()

    st.subheader("📈 Customer Sentiment Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "😊 Positive Reviews",
        f"{positive_percent}%"
    )

    col2.metric(
        "😐 Neutral Reviews",
        f"{neutral_percent}%"
    )

    col3.metric(
        "😡 Negative Reviews",
        f"{negative_percent}%"
    )


    # =====================================
    # BAR CHART
    # =====================================

    st.divider()

    st.subheader("📊 Sentiment Distribution")

    st.bar_chart(df['Sentiment'].value_counts())


    # =====================================
    # PIE CHART
    # =====================================

    fig, ax = plt.subplots(figsize=(2.8, 2.8))

    df['Sentiment'].value_counts().plot.pie(
        autopct='%1.1f%%',
        ax=ax,
        radius=0.65,
        textprops={'fontsize': 8}
    )

    ax.set_ylabel("")

    plt.tight_layout()

    st.subheader("🥧 Sentiment Distribution Pie Chart")

    st.pyplot(fig, use_container_width=False)


    # =====================================
    # WORD CLOUD
    # =====================================

    st.divider()

    st.subheader("☁️ Customer Review Word Cloud")

    text = " ".join(df['Text'].astype(str))

    wordcloud = WordCloud(
        width=500,
        height=250,
        background_color='white'
    ).generate(text)

    fig_wc, ax_wc = plt.subplots(figsize=(4.5, 2.2))

    ax_wc.imshow(wordcloud, interpolation='bilinear')

    ax_wc.axis("off")

    st.pyplot(fig_wc)


    # =====================================
    # AI INSIGHTS BUTTON
    # =====================================

    st.divider()

    if st.button("Generate AI Insights"):

        sample_reviews = " ".join(
            df['Text'].head(50).astype(str)
        )

        prompt = f"""
        Analyze these customer reviews.

        Give output in this format:

        Top 5 Customer Complaints:
        - complaint 1
        - complaint 2
        - complaint 3
        - complaint 4
        - complaint 5

        Top Positive Feedback Trends:
        - trend 1
        - trend 2
        - trend 3

        Business Improvement Suggestions:
        - suggestion 1
        - suggestion 2
        - suggestion 3

        Overall Customer Sentiment Summary:
        Provide a short summary.

        Customer Reviews:
        {sample_reviews}
        """


        # =================================
        # GEMINI PROCESSING
        # =================================

        with st.spinner(
            "🤖 Gemini AI is generating business insights..."
        ):

            try:

                response = gemini_model.generate_content(prompt)

                st.success("✅ AI Analysis Completed!")

                st.divider()

                st.subheader(
                    "🤖 AI Business Intelligence Report"
                )

                st.divider()

                st.markdown(response.text)


                # =================================
                # CREATE PROFESSIONAL PDF
                # =================================

                pdf = FPDF()

                pdf.set_auto_page_break(
                    auto=True,
                    margin=15
                )

                pdf.add_page()


                # =============================
                # HEADER
                # =============================

                pdf.set_fill_color(30, 30, 60)

                pdf.rect(0, 0, 210, 35, 'F')

                pdf.set_text_color(255, 255, 255)

                pdf.set_font("Arial", "B", 22)

                pdf.cell(
                    200,
                    18,
                    "AI Business Intelligence Report",
                    ln=True,
                    align="C"
                )

                pdf.set_font("Arial", "", 11)

                pdf.cell(
                    200,
                    5,
                    "Generated using Streamlit + Gemini AI",
                    ln=True,
                    align="C"
                )

                pdf.ln(18)


                # =============================
                # REPORT TITLE
                # =============================

                pdf.set_text_color(20, 20, 20)

                pdf.set_font("Arial", "B", 17)

                pdf.cell(
                    0,
                    10,
                    "Customer Review Analytics Summary",
                    ln=True
                )

                pdf.ln(3)

                pdf.set_draw_color(120, 120, 120)

                pdf.line(
                    10,
                    pdf.get_y(),
                    200,
                    pdf.get_y()
                )

                pdf.ln(10)


                # =============================
                # AI CONTENT
                # =============================

                safe_text = response.text.encode(
                    "latin-1",
                    "replace"
                ).decode("latin-1")

                sections = safe_text.split("\n")

                for line in sections:

                    line = line.strip()

                    # HEADINGS
                    if (
                        "Complaint" in line
                        or "Feedback" in line
                        or "Suggestions" in line
                        or "Summary" in line
                    ):

                        pdf.set_font("Arial", "B", 15)

                        pdf.set_text_color(220, 50, 50)

                        pdf.ln(5)

                        pdf.multi_cell(0,10,line)

                        pdf.ln(2)

                    # BULLET POINTS
                    elif line.startswith("-"):

                        pdf.set_font(
                            "Arial",
                            "",
                            12
                        )

                        pdf.set_text_color(40,40,40)

                        pdf.multi_cell(0, 8, f"- {line[1:]}")

                    # NORMAL TEXT
                    else:

                        pdf.set_font(
                            "Arial","",12)

                        pdf.set_text_color(0,0,0)

                        pdf.multi_cell(0,8,line)


                # =============================
                # FOOTER
                # =============================

                pdf.ln(15)

                pdf.set_draw_color(
                    150,
                    150,
                    150
                )

                pdf.line(10,pdf.get_y(),200,pdf.get_y())

                pdf.ln(8)

                pdf.set_font("Arial","I",10)

                pdf.set_text_color(100,100,100)

                pdf.cell(0,8,"AI Customer Review Intelligence System",ln=True,align="C")

                pdf.cell(0,8,"Made with Streamlit + Gemini AI",ln=True,align="C")


                # =============================
                # SAVE PDF
                # =============================

                pdf.output(
                    "AI_Business_Report.pdf"
                )


                # =============================
                # DOWNLOAD BUTTON
                # =============================

                with open(
                    "AI_Business_Report.pdf",
                    "rb"
                ) as file:

                    st.download_button(
                        label="📥 Download AI Business Report",
                        data=file,
                        file_name="AI_Business_Report.pdf",
                        mime="application/pdf"
                    )


            # =================================
            # ERROR HANDLING
            # =================================

            except Exception as e:

                st.error(
                    f"⚠️ Error occurred: {e}"
                )


# =========================================
# FOOTER
# =========================================

st.divider()

st.markdown("""
<center>

Made with ❤️ using Streamlit + Gemini AI

AI Customer Review Intelligence System

</center>
""", unsafe_allow_html=True)

