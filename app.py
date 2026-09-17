import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image


# 1. Folders for Storage
if not os.path.exists("found_db"):
    os.makedirs("found_db")

# 2. Database Creation using PANDAS DATAFRAME (Added contact_info structural column)
if 'lost_found_df' not in st.session_state:
    st.session_state.lost_found_df = pd.DataFrame(columns=['id', 'category', 'item_details', 'location', 'image_path', 'contact_info'])

st.title("🤝Lost & Found Intelligent Matcher")
# st.write("Project built using **Pandas DataFrames, Numpy, and Supervised KNN Algorithmic Math**")

tab1, tab2 = st.tabs(["🎒 Report Lost Item (User A)", "🔍 Report Found Item (User B)"])

# --- USER B: FOUND ITEM REGISTRATION ---
with tab2:
    st.header("Upload a Found Item")
    f_category = st.selectbox("Select Category", ["Laptop", "Mobile Phone", "Wallet", "Keys", "Others"], key="found_cat")
    f_name = st.text_input("Detailed Description (Brand, Color, Model):", placeholder="e.g., Black Dell Inspiron laptop")
    f_loc = st.text_input("Found Location:", placeholder="e.g., Library 2nd floor")
    
    # NEW SECURE BRIDGE FEATURE: Contact Information Field
    f_contact = st.text_input("Your Contact Info (Phone Number / Email ID):", placeholder="e.g., +91 9876543210 or finder@college.edu")
    
    f_file = st.file_uploader("Upload Actual Found Image", type=["jpg", "png"], key="found_upload")
    
    if st.button("Submit Found Report"):
        if f_name and f_file and f_contact:
            img_path = f"found_db/{f_file.name}"
            with open(img_path, "wb") as f:
                f.write(f_file.getbuffer())
            
            # Appending data into Pandas DataFrame using local dictionary logic
            df_length = len(st.session_state.lost_found_df)
            new_row = {
                'id': df_length + 1,
                'category': f_category,
                'item_details': f_name.lower().strip(),
                'location': f_loc,
                'image_path': img_path,
                'contact_info': f_contact.strip() # Store dynamic locator details
            }
            
            # Update dataframe
            st.session_state.lost_found_df = pd.concat([st.session_state.lost_found_df, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"Successfully added via Pandas DataFrame: '{f_name}'!")
        else:
            st.warning("Please provide description, contact info, and image.")

# --- USER A: LOST ITEM SEARCH ---
with tab1:
    st.header("Search Your Lost Item")
    l_category = st.selectbox("Select Lost Category", ["Laptop", "Mobile Phone", "Wallet", "Keys", "Others"], key="lost_cat")
    l_name = st.text_input("Describe your lost item specifically:", placeholder="e.g., Black Dell Laptop")
    
    if st.button("Run ML Search Engine"):
        if not l_name:
            st.error("Please enter a description.")
        elif st.session_state.lost_found_df.empty:
            st.info("The Pandas database is currently empty. No items to match.")
        else:
            # Filter Pandas Dataframe by Category (Supervised pre-filtering)
            current_df = st.session_state.lost_found_df
            filtered_df = current_df[current_df['category'] == l_category].reset_index(drop=True)
            
            if filtered_df.empty:
                st.error(f"No database records match the category '{l_category}'.")
            else:
                st.write("### Supervised Text Mining & KNN Distance Calculation Running...")
                
                # ---- PURE STRING MATH & NUMPY LOGIC (KNN Instance-Based Core) ----
                user_words = set(l_name.lower().split())
                found_any = False
                
                for idx in range(len(filtered_df)):
                    item = filtered_df.iloc[idx] # Fetching rows using Pandas iloc
                    db_words = item['item_details'].lower().split()
                    
                    # Feature Matching using Set Intersection
                    match_count = sum(1 for word in user_words if word in db_words)
                    total_unique_words = len(user_words.union(set(db_words)))
                    
                    # Distance Matrix Logic using Jaccard Similarity index (Syllabus Math)
                    if total_unique_words > 0:
                        similarity_score = match_count / total_unique_words
                        match_percentage = similarity_score * 100
                    else:
                        match_percentage = 0.0
                        
                    # Show match only if it passes a minimum threshold
                    if match_percentage > 10:
                        found_any = True
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if os.path.exists(item['image_path']):
                                img = Image.open(item['image_path'])
                                st.image(img, width=150)
                        with col2:
                            st.write(f"**Item:** {item['item_details'].title()}")
                            st.write(f"**Location:** {item['location']}")
                            st.metric(label="KNN Distance Matching Score", value=f"{match_percentage:.2f}%")
                            
                            # DYNAMIC CONTACT INFORMATION LOGIC IF MATCH IS SECURED
                            if match_percentage > 60:
                                st.success("🔥 High Probability Supervised Match!")
                                # Displaying information safely inside structural UI panel
                                st.info(f"📞 **Contact Finder to Retrieve:** {item['contact_info']}")
                            elif match_percentage > 30:
                                st.warning("⚠️ Partial Match Found.")
                                st.caption("Verify with the finder before retrieving.")
                        st.markdown("---")
                
                if not found_any:
                    st.error("No records matched your description features.")
