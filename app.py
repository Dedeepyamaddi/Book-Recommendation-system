import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/best_model.pkl"
    )

    return model


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    books = joblib.load(
        "models/books.pkl"
    )

    ratings = joblib.load(
        "models/final_ratings.pkl"
    )

    popular_books = joblib.load(
        "models/popular_books.pkl"
    )

    results = pd.read_csv(
        "models/model_results.csv"
    )

    return (
        books,
        ratings,
        popular_books,
        results
    )


model = load_model()

books, ratings, popular_books, results = load_data()


# ==========================================================
# CLEAN BOOK DATA
# ==========================================================

books = books.copy()

# Remove duplicate ISBN records
books = books.drop_duplicates(
    subset=["ISBN"]
)

# Fill missing book information
if "Book-Title" in books.columns:
    books["Book-Title"] = books["Book-Title"].fillna(
        "Unknown Book"
    )

if "Book-Author" in books.columns:
    books["Book-Author"] = books["Book-Author"].fillna(
        "Unknown Author"
    )

if "Publisher" in books.columns:
    books["Publisher"] = books["Publisher"].fillna(
        "Unknown Publisher"
    )

if "Year-Of-Publication" in books.columns:

    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"],
        errors="coerce"
    )

    books["Year-Of-Publication"] = books[
        "Year-Of-Publication"
    ].fillna(0)


# ==========================================================
# CREATE BOOK STATISTICS
# ==========================================================

book_statistics = ratings.groupby("ISBN").agg(
    Rating_Count=("Book-Rating", "count"),
    Average_Rating=("Book-Rating", "mean")
).reset_index()


# Merge statistics with books
books_with_stats = books.merge(
    book_statistics,
    on="ISBN",
    how="left"
)

books_with_stats["Rating_Count"] = (
    books_with_stats["Rating_Count"]
    .fillna(0)
)

books_with_stats["Average_Rating"] = (
    books_with_stats["Average_Rating"]
    .fillna(0)
)


# ==========================================================
# TITLE
# ==========================================================

st.title("📚 Book Recommendation System")

st.write(
    "An interactive collaborative filtering "
    "book recommendation system."
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Book Recommendation",
        "Model Performance",
        "EDA Visualizations"
    ]
)


# ==========================================================
# RECOMMENDATION FUNCTION
# ==========================================================

def recommend_books(
    user_id,
    model,
    ratings_data,
    books_data,
    candidate_books,
    n=10,
    min_rating=0,
    min_popularity=0,
    selected_author="All Authors",
    year_range=(0, 2025)
):

    # ------------------------------------------------------
    # Books already rated by user
    # ------------------------------------------------------

    user_rated_books = set(
        ratings_data[
            ratings_data["User-ID"] == user_id
        ]["ISBN"]
    )


    # ------------------------------------------------------
    # Candidate books
    # ------------------------------------------------------

    candidates = [
        isbn
        for isbn in candidate_books
        if isbn not in user_rated_books
    ]


    # ------------------------------------------------------
    # Book information
    # ------------------------------------------------------

    candidate_data = books_data[
        books_data["ISBN"].isin(candidates)
    ].copy()


    # ------------------------------------------------------
    # Apply author filter
    # ------------------------------------------------------

    if selected_author != "All Authors":

        candidate_data = candidate_data[
            candidate_data["Book-Author"]
            == selected_author
        ]


    # ------------------------------------------------------
    # Apply year filter
    # ------------------------------------------------------

    candidate_data = candidate_data[
        (
            candidate_data["Year-Of-Publication"]
            >= year_range[0]
        )
        &
        (
            candidate_data["Year-Of-Publication"]
            <= year_range[1]
        )
    ]


    # ------------------------------------------------------
    # Apply popularity filter
    # ------------------------------------------------------

    candidate_data = candidate_data[
        candidate_data["Rating_Count"]
        >= min_popularity
    ]


    candidates = candidate_data[
        "ISBN"
    ].tolist()


    # ------------------------------------------------------
    # Generate predictions
    # ------------------------------------------------------

    predictions = []

    for isbn in candidates:

        try:

            prediction = model.predict(
                user_id,
                isbn
            )

            predictions.append({
                "ISBN": isbn,
                "Predicted Rating": prediction.est
            })

        except Exception:
            continue


    if len(predictions) == 0:

        return pd.DataFrame()


    recommendation_df = pd.DataFrame(
        predictions
    )


    # ------------------------------------------------------
    # Merge book information
    # ------------------------------------------------------

    recommendation_df = recommendation_df.merge(
        books_data[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Publisher",
                "Year-Of-Publication",
                "Image-URL-M",
                "Rating_Count",
                "Average_Rating"
            ]
        ],
        on="ISBN",
        how="left"
    )


    # ------------------------------------------------------
    # Apply predicted rating filter
    # ------------------------------------------------------

    recommendation_df = recommendation_df[
        recommendation_df["Predicted Rating"]
        >= min_rating
    ]


    # ------------------------------------------------------
    # Sort by predicted rating
    # ------------------------------------------------------

    recommendation_df = (
        recommendation_df
        .sort_values(
            [
                "Predicted Rating",
                "Average_Rating",
                "Rating_Count"
            ],
            ascending=False
        )
        .head(n)
    )


    return recommendation_df


# ==========================================================
# PAGE 1 - BOOK RECOMMENDATION
# ==========================================================

if page == "Book Recommendation":

    st.header(
        "📖 Get Personalized Book Recommendations"
    )

    st.write(
        "Select a user and customize the recommendation "
        "criteria."
    )


    # ------------------------------------------------------
    # User selection
    # ------------------------------------------------------

    user_ids = sorted(
        ratings["User-ID"].unique()
    )

    selected_user = st.selectbox(
        "Select User ID",
        user_ids
    )


    # ------------------------------------------------------
    # Recommendation type
    # ------------------------------------------------------

    recommendation_type = st.selectbox(
        "Recommendation Type",
        [
            "Personalized Recommendations",
            "Highly Rated Books",
            "Popular Books"
        ]
    )


    # ------------------------------------------------------
    # Number of recommendations
    # ------------------------------------------------------

    number_books = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )


    # ------------------------------------------------------
    # Minimum predicted rating
    # ------------------------------------------------------

    min_rating = st.slider(
        "Minimum Predicted Rating",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5
    )


    # ------------------------------------------------------
    # Minimum popularity
    # ------------------------------------------------------

    min_popularity = st.slider(
        "Minimum Number of Ratings",
        min_value=0,
        max_value=50,
        value=0,
        step=1
    )


    # ------------------------------------------------------
    # Author selection
    # ------------------------------------------------------

    authors = sorted(
        books_with_stats[
            "Book-Author"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    selected_author = st.selectbox(
        "Filter by Author",
        ["All Authors"] + authors
    )


    # ------------------------------------------------------
    # Publication year
    # ------------------------------------------------------

    valid_years = books_with_stats[
        books_with_stats[
            "Year-Of-Publication"
        ] > 0
    ]["Year-Of-Publication"]

    if len(valid_years) > 0:

        min_year = int(
            valid_years.min()
        )

        max_year = int(
            valid_years.max()
        )

    else:

        min_year = 1900
        max_year = 2025


    year_range = st.slider(
        "Publication Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )


    # ------------------------------------------------------
    # Recommend button
    # ------------------------------------------------------

    if st.button(
        "📚 Recommend Books",
        use_container_width=True
    ):

        with st.spinner(
            "Generating recommendations..."
        ):

            # ==============================================
            # PERSONALIZED RECOMMENDATIONS
            # ==============================================

            if recommendation_type == (
                "Personalized Recommendations"
            ):

                recommendations = recommend_books(
                    selected_user,
                    model,
                    ratings,
                    books_with_stats,
                    popular_books,
                    number_books,
                    min_rating,
                    min_popularity,
                    selected_author,
                    year_range
                )


            # ==============================================
            # HIGHLY RATED BOOKS
            # ==============================================

            elif recommendation_type == (
                "Highly Rated Books"
            ):

                user_rated_books = set(
                    ratings[
                        ratings["User-ID"]
                        == selected_user
                    ]["ISBN"]
                )

                recommendations = (
                    books_with_stats[
                        ~books_with_stats["ISBN"]
                        .isin(user_rated_books)
                    ]
                )

                recommendations = recommendations[
                    recommendations["Average_Rating"]
                    >= min_rating
                ]

                recommendations = recommendations[
                    recommendations["Rating_Count"]
                    >= min_popularity
                ]

                if selected_author != "All Authors":

                    recommendations = recommendations[
                        recommendations["Book-Author"]
                        == selected_author
                    ]

                recommendations = recommendations[
                    (
                        recommendations[
                            "Year-Of-Publication"
                        ] >= year_range[0]
                    )
                    &
                    (
                        recommendations[
                            "Year-Of-Publication"
                        ] <= year_range[1]
                    )
                ]

                recommendations = (
                    recommendations
                    .sort_values(
                        [
                            "Average_Rating",
                            "Rating_Count"
                        ],
                        ascending=False
                    )
                    .head(number_books)
                )

                recommendations[
                    "Predicted Rating"
                ] = recommendations[
                    "Average_Rating"
                ]


            # ==============================================
            # POPULAR BOOKS
            # ==============================================

            else:

                user_rated_books = set(
                    ratings[
                        ratings["User-ID"]
                        == selected_user
                    ]["ISBN"]
                )

                recommendations = (
                    books_with_stats[
                        ~books_with_stats["ISBN"]
                        .isin(user_rated_books)
                    ]
                )

                recommendations = recommendations[
                    recommendations["Rating_Count"]
                    >= min_popularity
                ]

                if selected_author != "All Authors":

                    recommendations = recommendations[
                        recommendations["Book-Author"]
                        == selected_author
                    ]

                recommendations = recommendations[
                    (
                        recommendations[
                            "Year-Of-Publication"
                        ] >= year_range[0]
                    )
                    &
                    (
                        recommendations[
                            "Year-Of-Publication"
                        ] <= year_range[1]
                    )
                ]

                recommendations = (
                    recommendations
                    .sort_values(
                        [
                            "Rating_Count",
                            "Average_Rating"
                        ],
                        ascending=False
                    )
                    .head(number_books)
                )

                recommendations[
                    "Predicted Rating"
                ] = recommendations[
                    "Average_Rating"
                ]


        # ==================================================
        # DISPLAY RESULTS
        # ==================================================

        if recommendations.empty:

            st.warning(
                "No books found with the selected filters. "
                "Try reducing the minimum rating, popularity "
                "or changing the author/year filters."
            )

        else:

            st.success(
                f"Found {len(recommendations)} "
                "recommended books."
            )

            st.subheader(
                "📚 Recommended Books"
            )


            # ------------------------------------------------
            # Summary metrics
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Books Recommended",
                    len(recommendations)
                )

            with col2:

                st.metric(
                    "Average Rating",
                    f"{recommendations['Average_Rating'].mean():.2f}"
                )

            with col3:

                st.metric(
                    "Average Predicted Rating",
                    f"{recommendations['Predicted Rating'].mean():.2f}"
                )


            # ------------------------------------------------
            # Display books
            # ------------------------------------------------

            for index, row in recommendations.iterrows():

                col1, col2 = st.columns(
                    [1, 4]
                )

                with col1:

                    image_url = row.get(
                        "Image-URL-M",
                        ""
                    )

                    if (
                        pd.notna(image_url)
                        and str(image_url).strip() != ""
                    ):

                        try:

                            st.image(
                                image_url,
                                width=140
                            )

                        except Exception:

                            st.write(
                                "📕 No Image"
                            )

                    else:

                        st.write(
                            "📕 No Image"
                        )


                with col2:

                    st.markdown(
                        f"### 📖 {row['Book-Title']}"
                    )

                    st.write(
                        f"**Author:** "
                        f"{row['Book-Author']}"
                    )

                    st.write(
                        f"**Publisher:** "
                        f"{row['Publisher']}"
                    )

                    year = row[
                        "Year-Of-Publication"
                    ]

                    if year != 0:

                        st.write(
                            f"**Publication Year:** "
                            f"{int(year)}"
                        )

                    st.write(
                        f"⭐ **Predicted Rating:** "
                        f"{row['Predicted Rating']:.2f}"
                    )

                    st.write(
                        f"📊 **Average Rating:** "
                        f"{row['Average_Rating']:.2f}"
                    )

                    st.write(
                        f"👥 **Number of Ratings:** "
                        f"{int(row['Rating_Count'])}"
                    )

                    st.divider()


# ==========================================================
# PAGE 2 - MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.header(
        "🤖 Model Performance"
    )


    st.dataframe(
        results.sort_values(
            "RMSE"
        ),
        use_container_width=True
    )


    best_model_name = results.loc[
        results["RMSE"].idxmin(),
        "Model"
    ]

    best_rmse = results[
        "RMSE"
    ].min()

    best_mae = results.loc[
        results["RMSE"].idxmin(),
        "MAE"
    ]


    st.success(
        f"🏆 Best Model: {best_model_name}"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Best RMSE",
            f"{best_rmse:.4f}"
        )


    with col2:

        st.metric(
            "Best MAE",
            f"{best_mae:.4f}"
        )


    st.subheader(
        "Model Comparison"
    )


    st.bar_chart(
        results.set_index(
            "Model"
        )["RMSE"]
    )


# ==========================================================
# PAGE 3 - EDA VISUALIZATIONS
# ==========================================================

elif page == "EDA Visualizations":

    st.header(
        "📊 Exploratory Data Analysis"
    )


    visualization_files = {

        "Rating Distribution":
            "visualizations/rating_distribution.png",

        "Age Distribution":
            "visualizations/age_distribution.png",

        "Top Books":
            "visualizations/top_books.png",

        "Top Authors":
            "visualizations/top_authors.png",

        "Ratings by Year":
            "visualizations/ratings_by_year.png",

        "User Activity":
            "visualizations/user_activity.png",

        "Book Popularity":
            "visualizations/book_popularity.png",

        "Model Comparison":
            "visualizations/model_comparison.png"
    }


    for title, file_path in visualization_files.items():

        st.subheader(title)


        if os.path.exists(file_path):

            st.image(
                file_path,
                use_container_width=True
            )

        else:

            st.warning(
                f"{file_path} not found."
            )