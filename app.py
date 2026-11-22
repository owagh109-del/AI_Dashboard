import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Dynamic Pricing Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.sub-title {
    text-align: center;
    font-size: 19px;
    margin-bottom: 30px;
}

.price-box {
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #4CAF50;
    background-color: #f0fff4;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">💰 Smart Dynamic Pricing Engine</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-powered pricing optimization for local e-commerce businesses'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GENERATE DEMO E-COMMERCE DATA
# ============================================================

@st.cache_data
def generate_data():

    np.random.seed(42)

    n = 1500

    data = pd.DataFrame({

        "Day": np.random.randint(1, 31, n),

        "Product_ID": np.random.randint(
            1001, 1021, n
        ),

        "Base_Price": np.random.uniform(
            100, 5000, n
        ),

        "Inventory": np.random.randint(
            5, 500, n
        ),

        "Competitor_Price": np.random.uniform(
            100, 5000, n
        ),

        "Customer_Visits": np.random.randint(
            20, 2000, n
        ),

        "Previous_Sales": np.random.randint(
            1, 300, n
        ),

        "Rating": np.random.uniform(
            2.5, 5.0, n
        ),

        "Discount": np.random.uniform(
            0, 40, n
        ),

        "Weekend": np.random.randint(
            0, 2, n
        ),

        "Festival": np.random.randint(
            0, 2, n
        )
    })


    # --------------------------------------------------------
    # SIMULATED DEMAND
    # --------------------------------------------------------

    demand_factor = (

        0.0008 * data["Customer_Visits"]

        + 0.5 * data["Previous_Sales"]

        + 10 * data["Weekend"]

        + 20 * data["Festival"]

        + 2 * data["Discount"]

        + np.random.normal(0, 10, n)
    )

    data["Demand"] = demand_factor.clip(lower=1)


    # --------------------------------------------------------
    # ACTUAL SALES
    # --------------------------------------------------------

    data["Sales"] = (

        data["Demand"]
        * np.random.uniform(0.7, 1.1, n)

    ).clip(lower=1)


    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    data["Revenue"] = (

        data["Sales"]
        * data["Base_Price"]
        * (1 - data["Discount"] / 100)

    )


    # --------------------------------------------------------
    # PRODUCT CATEGORY
    # --------------------------------------------------------

    categories = [
        "Electronics",
        "Clothing",
        "Grocery",
        "Home & Kitchen",
        "Beauty",
        "Accessories"
    ]

    data["Category"] = np.random.choice(
        categories,
        n
    )

    return data


df = generate_data()


# ============================================================
# MACHINE LEARNING
# ============================================================

features = [

    "Day",
    "Base_Price",
    "Inventory",
    "Competitor_Price",
    "Customer_Visits",
    "Previous_Sales",
    "Rating",
    "Discount",
    "Weekend",
    "Festival"
]


X = df[features]

y = df["Demand"]


# ============================================================
# TRAIN MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42
)


model = RandomForestRegressor(

    n_estimators=150,

    random_state=42
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# MODEL EVALUATION
# ============================================================

prediction = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    prediction
)


# ============================================================
# DYNAMIC PRICING FUNCTION
# ============================================================

def calculate_dynamic_price(

    base_price,
    competitor_price,
    inventory,
    predicted_demand,
    rating,
    discount

):

    price = base_price


    # --------------------------------------------------------
    # HIGH DEMAND
    # --------------------------------------------------------

    if predicted_demand > 150:

        price *= 1.10

    elif predicted_demand > 100:

        price *= 1.05


    # --------------------------------------------------------
    # LOW INVENTORY
    # --------------------------------------------------------

    if inventory < 30:

        price *= 1.08

    elif inventory < 60:

        price *= 1.04


    # --------------------------------------------------------
    # HIGH INVENTORY
    # --------------------------------------------------------

    if inventory > 300:

        price *= 0.90

    elif inventory > 200:

        price *= 0.95


    # --------------------------------------------------------
    # COMPETITOR PRICE
    # --------------------------------------------------------

    if competitor_price < price:

        price *= 0.97

    elif competitor_price > price * 1.10:

        price *= 1.03


    # --------------------------------------------------------
    # PRODUCT RATING
    # --------------------------------------------------------

    if rating >= 4.5:

        price *= 1.02

    elif rating < 3.5:

        price *= 0.95


    # --------------------------------------------------------
    # EXISTING DISCOUNT
    # --------------------------------------------------------

    if discount > 30:

        price *= 0.97


    # --------------------------------------------------------
    # PRICE LIMIT
    # --------------------------------------------------------

    minimum_price = base_price * 0.70

    maximum_price = base_price * 1.30

    price = max(
        minimum_price,
        min(price, maximum_price)
    )


    return round(price, 2)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📋 Dashboard Menu")


page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Dashboard",

        "📦 Product Analytics",

        "🤖 AI Demand Model",

        "💰 Dynamic Pricing",

        "🔮 Price Simulator",

        "📈 Sales Analysis",

        "💡 Recommendations"

    ]

)


st.sidebar.markdown("---")


st.sidebar.info("""
### 🎯 Real-World Problem

Local e-commerce businesses often use fixed prices even when demand,
competition and inventory change.

### 🤖 AI Solution

The system predicts demand and automatically recommends an optimized price.

### 💰 Goal

Increase revenue while keeping prices competitive and managing inventory.
""")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "🏠 E-Commerce Pricing Dashboard"
    )


    st.write("""
    The Smart Dynamic Pricing Engine uses Machine Learning,
    inventory information, customer demand and competitor pricing
    to recommend better product prices.
    """)


    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_sales = df["Sales"].sum()

    total_revenue = df["Revenue"].sum()

    average_price = df["Base_Price"].mean()

    average_demand = df["Demand"].mean()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "🛒 Total Sales",

            f"{total_sales:,.0f}"

        )


    with col2:

        st.metric(

            "💰 Total Revenue",

            f"₹{total_revenue:,.0f}"

        )


    with col3:

        st.metric(

            "🏷️ Average Price",

            f"₹{average_price:,.0f}"

        )


    with col4:

        st.metric(

            "📈 Average Demand",

            f"{average_demand:,.1f}"

        )


    st.markdown("---")


    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    st.subheader(
        "🔄 Smart Pricing Process"
    )


    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.info(
            "1️⃣\n\nCollect Data"
        )


    with col2:

        st.info(
            "2️⃣\n\nPredict Demand"
        )


    with col3:

        st.info(
            "3️⃣\n\nAnalyze Competition"
        )


    with col4:

        st.info(
            "4️⃣\n\nOptimize Price"
        )


    with col5:

        st.info(
            "5️⃣\n\nIncrease Revenue"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # DAILY REVENUE
    # --------------------------------------------------------

    st.subheader(
        "📈 Daily Revenue"
    )


    daily_revenue = df.groupby(
        "Day"
    )["Revenue"].sum()


    fig, ax = plt.subplots(
        figsize=(11, 4)
    )


    ax.plot(

        daily_revenue.index,

        daily_revenue.values,

        marker="o"

    )


    ax.set_xlabel(
        "Day"
    )


    ax.set_ylabel(
        "Revenue (₹)"
    )


    ax.set_title(
        "Daily Revenue"
    )


    ax.grid()


    st.pyplot(fig)


# ============================================================
# PRODUCT ANALYTICS
# ============================================================

elif page == "📦 Product Analytics":

    st.header(
        "📦 Product Analytics"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Products",

            df["Product_ID"].nunique()

        )


    with col2:

        st.metric(

            "Average Inventory",

            f"{df['Inventory'].mean():.0f}"

        )


    with col3:

        st.metric(

            "Average Rating",

            f"{df['Rating'].mean():.2f}"

        )


    st.markdown("---")


    st.subheader(
        "📊 Product Dataset"
    )


    st.dataframe(

        df,

        use_container_width=True

    )


    st.markdown("---")


    st.subheader(
        "📦 Inventory Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.hist(
        df["Inventory"],
        bins=30
    )


    ax.set_xlabel(
        "Inventory"
    )


    ax.set_ylabel(
        "Number of Products"
    )


    ax.set_title(
        "Inventory Distribution"
    )


    st.pyplot(fig)


# ============================================================
# AI MODEL
# ============================================================

elif page == "🤖 AI Demand Model":

    st.header(
        "🤖 AI Demand Prediction Model"
    )


    st.write("""
    The Random Forest model predicts expected product demand
    using customer visits, previous sales, inventory, pricing,
    discounts and seasonal factors.
    """)


    st.metric(

        "Mean Absolute Error",

        f"{mae:.2f} units"

    )


    st.markdown("---")


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "⭐ Demand Factors"
    )


    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance": model.feature_importances_

    }).sort_values(

        "Importance",

        ascending=False

    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.barh(

        importance_df["Feature"],

        importance_df["Importance"]

    )


    ax.set_xlabel(
        "Importance"
    )


    ax.set_title(
        "AI Feature Importance"
    )


    ax.invert_yaxis()


    st.pyplot(fig)


    st.dataframe(

        importance_df,

        use_container_width=True

    )


# ============================================================
# DYNAMIC PRICING
# ============================================================

elif page == "💰 Dynamic Pricing":

    st.header(
        "💰 Dynamic Pricing Engine"
    )


    st.write("""
    This section analyzes current market conditions and generates
    an AI-based recommended selling price.
    """)


    product = st.selectbox(

        "Select Product",

        sorted(
            df["Product_ID"].unique()
        )

    )


    product_data = df[
        df["Product_ID"] == product
    ].iloc[0]


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "Base Price",

            f"₹{product_data['Base_Price']:,.0f}"

        )


    with col2:

        st.metric(

            "Competitor Price",

            f"₹{product_data['Competitor_Price']:,.0f}"

        )


    with col3:

        st.metric(

            "Inventory",

            f"{product_data['Inventory']:,.0f}"

        )


    with col4:

        st.metric(

            "Rating",

            f"{product_data['Rating']:.1f} ⭐"

        )


    # --------------------------------------------------------
    # DEMAND PREDICTION
    # --------------------------------------------------------

    input_row = pd.DataFrame([{

        feature: product_data[feature]

        for feature in features

    }])


    predicted_demand = model.predict(
        input_row
    )[0]


    recommended_price = calculate_dynamic_price(

        product_data["Base_Price"],

        product_data["Competitor_Price"],

        product_data["Inventory"],

        predicted_demand,

        product_data["Rating"],

        product_data["Discount"]

    )


    st.markdown("---")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "🤖 Predicted Demand",

            f"{predicted_demand:.0f} units"

        )


    with col2:

        st.metric(

            "🏷️ Recommended Price",

            f"₹{recommended_price:,.2f}"

        )


    with col3:

        price_change = (

            (recommended_price
             - product_data["Base_Price"])

            /

            product_data["Base_Price"]

            * 100

        )


        st.metric(

            "📊 Price Change",

            f"{price_change:+.2f}%"

        )


    st.markdown("---")


    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    st.subheader(
        "🧠 Pricing Decision"
    )


    if predicted_demand > 150:

        st.success(
            "🔥 High demand detected. "
            "A moderate price increase may be appropriate."
        )

    elif predicted_demand < 50:

        st.warning(
            "📉 Low demand detected. "
            "Consider a discount to increase sales."
        )

    else:

        st.info(
            "⚖️ Normal demand detected. "
            "Keep the price near the current market level."
        )


    if product_data["Inventory"] < 30:

        st.warning(
            "⚠️ Inventory is very low."
        )

    elif product_data["Inventory"] > 300:

        st.warning(
            "📦 Inventory is high. "
            "Consider reducing the price."
        )


# ============================================================
# PRICE SIMULATOR
# ============================================================

elif page == "🔮 Price Simulator":

    st.header(
        "🔮 AI Price Simulator"
    )


    st.write("""
    Enter current market conditions and see what price
    the system recommends.
    """)


    col1, col2 = st.columns(2)


    with col1:

        base_price = st.number_input(

            "🏷️ Base Price (₹)",

            min_value=10.0,

            max_value=100000.0,

            value=1000.0

        )


        competitor_price = st.number_input(

            "🏪 Competitor Price (₹)",

            min_value=10.0,

            max_value=100000.0,

            value=950.0

        )


        inventory = st.number_input(

            "📦 Current Inventory",

            min_value=1,

            max_value=10000,

            value=100

        )


        rating = st.slider(

            "⭐ Product Rating",

            1.0,

            5.0,

            4.0

        )


    with col2:

        visits = st.number_input(

            "👥 Customer Visits",

            min_value=1,

            max_value=10000,

            value=500

        )


        previous_sales = st.number_input(

            "🛒 Previous Sales",

            min_value=1,

            max_value=2000,

            value=100

        )


        discount = st.slider(

            "🏷️ Current Discount (%)",

            0.0,

            80.0,

            10.0

        )


        weekend = st.selectbox(

            "📅 Weekend?",

            ["No", "Yes"]

        )


        festival = st.selectbox(

            "🎉 Festival / Special Event?",

            ["No", "Yes"]

        )


    weekend_value = 1 if weekend == "Yes" else 0

    festival_value = 1 if festival == "Yes" else 0


    st.markdown("---")


    if st.button(

        "🤖 CALCULATE SMART PRICE",

        use_container_width=True

    ):


        input_data = pd.DataFrame([{

            "Day": 15,

            "Base_Price": base_price,

            "Inventory": inventory,

            "Competitor_Price": competitor_price,

            "Customer_Visits": visits,

            "Previous_Sales": previous_sales,

            "Rating": rating,

            "Discount": discount,

            "Weekend": weekend_value,

            "Festival": festival_value

        }])


        predicted_demand = model.predict(
            input_data
        )[0]


        recommended_price = calculate_dynamic_price(

            base_price,

            competitor_price,

            inventory,

            predicted_demand,

            rating,

            discount

        )


        estimated_revenue = (

            predicted_demand
            * recommended_price

        )


        current_revenue = (

            predicted_demand
            * base_price

        )


        revenue_difference = (

            estimated_revenue
            - current_revenue

        )


        st.markdown("---")


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(

                "🤖 Predicted Demand",

                f"{predicted_demand:.0f} units"

            )


        with col2:

            st.metric(

                "💰 Recommended Price",

                f"₹{recommended_price:,.2f}"

            )


        with col3:

            st.metric(

                "📈 Expected Revenue",

                f"₹{estimated_revenue:,.0f}"

            )


        with col4:

            st.metric(

                "💹 Revenue Difference",

                f"₹{revenue_difference:,.0f}"

            )


        st.markdown("---")


        if recommended_price > base_price:

            st.success(
                f"""
                📈 **Price Increase Recommended**

                Current Price: ₹{base_price:,.2f}

                Recommended Price: ₹{recommended_price:,.2f}

                Reason: Demand and/or market conditions support
                a moderate price increase.
                """
            )


        elif recommended_price < base_price:

            st.warning(
                f"""
                📉 **Price Reduction Recommended**

                Current Price: ₹{base_price:,.2f}

                Recommended Price: ₹{recommended_price:,.2f}

                Reason: Lower demand, high inventory or competitive
                pricing suggests reducing the price.
                """
            )


        else:

            st.info(
                "⚖️ Current price is suitable for the predicted market conditions."
            )


# ============================================================
# SALES ANALYSIS
# ============================================================

elif page == "📈 Sales Analysis":

    st.header(
        "📈 Sales & Revenue Analysis"
    )


    # --------------------------------------------------------
    # DAILY SALES
    # --------------------------------------------------------

    daily_sales = df.groupby(
        "Day"
    )["Sales"].sum()


    st.subheader(
        "🛒 Daily Sales"
    )


    fig, ax = plt.subplots(
        figsize=(11, 5)
    )


    ax.bar(

        daily_sales.index,

        daily_sales.values

    )


    ax.set_xlabel(
        "Day"
    )


    ax.set_ylabel(
        "Sales"
    )


    ax.set_title(
        "Daily Product Sales"
    )


    st.pyplot(fig)


    # --------------------------------------------------------
    # CATEGORY REVENUE
    # --------------------------------------------------------

    st.subheader(
        "💰 Revenue by Category"
    )


    category_revenue = df.groupby(
        "Category"
    )["Revenue"].sum()


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.bar(

        category_revenue.index,

        category_revenue.values

    )


    ax.set_xlabel(
        "Category"
    )


    ax.set_ylabel(
        "Revenue (₹)"
    )


    ax.set_title(
        "Revenue by Product Category"
    )


    plt.xticks(
        rotation=30
    )


    st.pyplot(fig)


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.header(
        "💡 Smart Pricing Recommendations"
    )


    st.write("""
    AI-based recommendations help local sellers decide when
    to increase, decrease or maintain product prices.
    """)


    recommendations = [

        "📈 Increase prices moderately when demand is very high.",

        "📉 Offer discounts when demand is low.",

        "📦 Reduce prices for products with excessive inventory.",

        "⚠️ Monitor products with critically low inventory.",

        "🏪 Compare prices with local competitors.",

        "⭐ Highly-rated products may support a small price premium.",

        "🎉 Adjust prices during festivals and special events.",

        "📊 Use historical sales data to predict future demand.",

        "💰 Optimize prices for revenue instead of using fixed pricing.",

        "🔄 Recalculate recommended prices regularly as market conditions change."

    ]


    for recommendation in recommendations:

        st.write(
            "✅ " + recommendation
        )


    st.markdown("---")


    st.success("""
    🎯 PROJECT GOAL

    Help local e-commerce businesses make smarter pricing decisions,
    increase revenue, manage inventory and remain competitive.
    """)


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📥 Export Data"
)


csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.sidebar.download_button(

    label="📥 Download Sales Dataset",

    data=csv_data,

    file_name="ecommerce_pricing_data.csv",

    mime="text/csv"

)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center">

    <b>💰 Smart Dynamic Pricing Engine for Local E-Commerce</b>

    <br>

    Artificial Intelligence • Machine Learning • Data Analytics

    <br>

    Real-World Problem Solving Project

    </div>
    """,
    unsafe_allow_html=True
)