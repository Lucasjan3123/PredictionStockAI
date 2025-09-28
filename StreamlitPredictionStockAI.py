import streamlit as st
import PredictionStockAI as stock_ai


main_page = st.Page(stock_ai.run_stock_prediction, title="Stock Prediction", icon="🎈")
page_2 = st.Page("CompanyRecordAnalyzer.py", title="Company Record Analyzer", icon="❄️")
page_3 = st.Page("AIAdvisorySystem.py", title="AI Advisory System", icon="🎉")

# Set up navigation
pg = st.navigation([main_page,page_2, page_3])

# Run the selected page
pg.run()