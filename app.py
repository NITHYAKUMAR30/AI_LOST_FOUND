import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# 1. Folders for Storage
if not os.path.exists("found_db"):
    os.makedirs("found_db")

# 2. Database Creation using PANDAS DATAFRAME
if 'lost_found_df' not in st.session_state:
    st.session_state.lost_found_df = pd.DataFrame(
        columns=[
            'id',
            'category',
            'item_details',
            'location',
            'image_path',
            'contact_info'
        ]
    )

st.title("🤝Lost & Found Intelligent Matcher")

tab1, tab2 = st.tabs(
    ["🎒 Report Lost Item (User A)", "🔍 Report Found Item (User B)"]
)

# --- USER B: FOUND ITEM REGISTRATION ---
with tab2:

    st.header("Upload a Found Item")

    f_category = st.selectbox(
        "Select Category",
        ["Laptop", "Mobile Phone", "Wallet", "Keys", "Others"],
        key="found_cat"
    )

    f_name = st.text_input(
        "Detailed Description (Brand, Color, Model):",
        placeholder="e.g., Black Dell Inspiron laptop"
    )

    f_loc = st.text_input(
        "Found Location:",
        placeholder="e.g., Library 2nd floor"
    )

    f_contact = st.text_input(
        "Your Contact Info (Phone Number / Email ID):",
        placeholder="e.g., +91 9876543210 or finder@college.edu"
    )

    f_file = st.file_uploader(
        "Upload Actual Found Image",
        type=["jpg", "png"],
        key="found_upload"
    )

    if st.button("Submit Found Report"):

        if f_name and f_file and f_contact:

            img_path = f"found_db/{f_file.name}"

            with open(img_path, "wb") as f:
                f.write(f_file.getbuffer())

            df_length = len(st.session_state.lost_found_df)

            new_row = {
                'id': df_length + 1,
                'category': f_category,
                'item_details': f_name.lower().strip(),
                'location': f_loc,
                'image_path': img_path,
                'contact_info': f_contact.strip()
            }

            st.session_state.lost_found_df = pd.concat(
                [
                    st.session_state.lost_found_df,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            st.success(
                f"Successfully added via Pandas DataFrame: '{f_name}'!"
            )

        else:

            st.warning(
                "Please provide description, contact info, and image."
            )


# --- USER A: LOST ITEM SEARCH ---
with tab1:

    st.header("Search Your Lost Item")

    l_category = st.selectbox(
        "Select Lost Category",
        ["Laptop", "Mobile Phone", "Wallet", "Keys", "Others"],
        key="lost_cat"
    )

    l_name = st.text_input(
        "Describe your lost item specifically:",
        placeholder="e.g., Black Dell Laptop"
    )

    if st.button("Run ML Search Engine"):

        if not l_name:

            st.error("Please enter a description.")

        elif st.session_state.lost_found_df.empty:

            st.info(
                "The Pandas database is currently empty. No items to match."
            )

        else:

            # ------------------------------------
            # STEP 5: CATEGORY FILTERING
            # ------------------------------------

            current_df = st.session_state.lost_found_df

            filtered_df = current_df[
                current_df['category'] == l_category
            ].reset_index(drop=True)

            if filtered_df.empty:

                st.error(
                    f"No database records match the category '{l_category}'."
                )

            else:

                st.write(
                    "### KNN Text Matching Running..."
                )

                # ------------------------------------
                # STEP 6: KNN TEXT MATCHING
                # ------------------------------------

                db_descriptions = filtered_df[
                    'item_details'
                ].tolist()

                all_descriptions = [
                    l_name.lower().strip()
                ] + db_descriptions

                vectorizer = TfidfVectorizer()

                feature_matrix = vectorizer.fit_transform(
                    all_descriptions
                )

                user_vector = feature_matrix[0:1]

                db_vectors = feature_matrix[1:]

                k = min(3, len(filtered_df))

                knn = NearestNeighbors(
                    n_neighbors=k,
                    metric='cosine'
                )

                knn.fit(db_vectors)

                distances, indices = knn.kneighbors(
                    user_vector
                )

                found_any = False

                # ------------------------------------
                # STEP 7: DISPLAY MATCHING RESULTS
                # ------------------------------------

                for distance, idx in zip(
                    distances[0],
                    indices[0]
                ):

                    item = filtered_df.iloc[idx]

                    # Adjusted matching score
                    tfidf_similarity = max(0, 1 - distance)

                    query_words = set(
                        l_name.lower().strip().split()
                    )

                    item_words = set(
                        item['item_details'].lower().strip().split()
                    )

                    common_words = query_words.intersection(
                        item_words
                    )

                    word_overlap = len(common_words) / max(
                        len(query_words),
                        len(item_words)
                    )

                    match_percentage = (
                        tfidf_similarity * word_overlap * 100
                    )

                    # Show only matches above 10%
                    if match_percentage > 10:

                        found_any = True

                        col1, col2 = st.columns(2)

                        with col1:

                            if os.path.exists(item['image_path']):

                                img = Image.open(
                                    item['image_path']
                                )

                                st.image(
                                    img,
                                    width=150
                                )

                        with col2:

                            st.write(
                                f"**Item:** "
                                f"{item['item_details'].title()}"
                            )

                            st.write(
                                f"**Location:** "
                                f"{item['location']}"
                            )

                            st.metric(
                                label="KNN Similarity Score",
                                value=f"{match_percentage:.2f}%"
                            )

                            # ------------------------------------
                            # STEP 8: CONTACT INFORMATION
                            # ------------------------------------

                            if match_percentage > 75:

                                st.success(
                                    "High Probability Match!"
                                )

                                st.info(
                                    f"**Contact Finder:** "
                                    f"{item['contact_info']}"
                                )

                            elif match_percentage > 30:

                                st.warning(
                                    "Partial Match Found."
                                )

                                st.caption(
                                    "Verify with the finder before retrieving."
                                )

                        st.markdown("---")

                # No matching results
                if not found_any:

                    st.error(
                        "No records matched your description features."
                    )
