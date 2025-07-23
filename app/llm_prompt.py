
from langchain_google_genai import ChatGoogleGenerativeAI
from app.env import import_my_env
from app.data_cache import forecast_data

import_my_env()
#GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

def build_structured_prompt(user_input, stock_summaries: dict):
    prompt = "[INST]<<SYS>>"
    prompt += (
        "You are a financial advisor. Your task is to recommend 5–10 stocks based on provided User profile, information and analyze recent inforamtion from online news, company finatials, recent stock trends quarterly results etc"
        "For each selected stock, provide ONLY: (1) one-line summary (e.g. good growth, or strong sentiment), (2) one-line concern (e.g. high P/E, or weak margins), and (3) Expected growth percentage (4) percentage to invest.(e.g Invest: 10%)"
        "Keep each stock's section brief (max 2 lines + %)."
        "At the end add Cautionary banner"
        "Caution: The following response is generated based on your\
         current financial inputs and historical data patterns.\
         It is intended solely for informational purposes. \
         The stock market involves inherent risk, and past performance \
         does not guarantee future results. This is not personalized financial advice.\
         Please consult a certified financial advisor before making any investment decisions."
        "\nHere is an reference table format. Make sure to select table format such that it should be properly displayed on UI:\n"
        """
        |       | **Stock**  | **Summary**                                       | **Concern**              | **Growth (%)** | **Allocation (%)** |
        | ----: | ---------- | ------------------------------------------------- | ------------------------ | -------------- | ------------------ |
        |     1 | DCMSHRIRAM | Strong revenue growth amidst challenges.          | High P/E ratio           | 21.97%         | 15%                |
        |     2 | LAURUSLABS | High insider ownership suggests confidence.       | Extremely high P/E ratio | 22.46%         | 15%                |
        |     3 | RBLBANK    | Navigating growth amidst challenges.              | High P/E ratio           | 19.34%         | 15%                |
        |     4 | AMBER      | Record revenue and growth.                        | Very high P/E ratio      | 18.53%         | 10%                |
        |     5 | RELAXO     | Navigating challenges with strategic initiatives. | High P/E ratio           | 16.79%         | 10%                |
        |     6 | AUBANK     | Strong deposit growth.                            | Moderate P/E ratio       | 15.98%         | 10%                |
        |     7 | GMMPFAUDLR | Strong cash flow and India growth.                | Very high P/E ratio      | 14.75%         | 10%                |
        |     8 | NATCOPHARM | Record profits amid future growth.                | Lower growth vs others   | 14.65%         | 15%                |
        """
        "\n<</SYS>>\n"
    )

    for symbol, summary in stock_summaries.items():
        growth_pct = forecast_data.get(symbol, 0.0) * 100
        prompt += f"[{symbol} Intro]: {summary['intro'][:200]}\n"
        prompt += f"[{symbol} Growth]: Forecasted Growth ~ {growth_pct:.2f}%\n"
        prompt += f"{summary['movement']}\n"

        prompt += "News:\n"
        for h in summary.get("headlines", [])[:1]:
            prompt += f"- {h}\n"

        prompt += "Financials:\n"
        for line in summary.get("financials", [])[:2]:
            prompt += f"- {line}\n"

    prompt += "Please recommend the top stocks. Keep your answers concise and actionable."
    prompt += "[/INST]"
    return prompt

def get_final_stock_advice(user_input: dict, stock_summaries: dict) -> tuple[str, dict]:

    print("########get_final_stock_advice")
    print("#### stock_summaries ####\n", stock_summaries)
    prompt = build_structured_prompt(user_input, stock_summaries)
    #print("\n###### Prompt - Start#####\n")
    #print(prompt)
    #print("\n###### Prompt - End  #####\n")
    result = llm.invoke(prompt)
    #print(f"\n###### result.content  #####\n {result.content}")
    return result.content, stock_summaries
