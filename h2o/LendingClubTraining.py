#!/usr/bin/env python
# coding: utf-8

# # *Lending Club Training*
# 
# In this tutorial, we will go through a step-by-step workflow to determine loan deliquency.  Predictions are made based on the information available at the time the loan was issued.  Our data is a portion of the public Lending Club dataset.

# ## Workflow
# 
# 1. Start H2O-3 cluster
# 2. Import data
# 3. Clean data
# 4. Feature engineering
# 5. Model training
# 6. Examine model accuracy
# 7. Interpret model
# 8. Stop H2O-3 cluster

# # Step 1 (of 8). Start H2O-3 cluster

# In[92]:


import h2o
import sys
h2o.init(url=f"http://{sys.argv[1]}:54321")


# # Step 2 (of 8). Import data
# 
# ### View Data

# In[93]:


data_path ="https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/lending-club/LoanStats3a.csv"
loans = h2o.import_file(data_path,
                        col_types = {"int_rate":"string", 
                                     "revol_util":"string", 
                                     "emp_length":"string", 
                                     "verification_status":"string"})


# In[94]:


loans.dim


# In[95]:


loans.head()


# ### Filter Loans
# 
# Now we will filter out loans that are ongoing.  These are loans with loan status like "Current" and "In Grace Period".

# In[96]:


num_unfiltered_loans = loans.dim[0]
num_unfiltered_loans


# In[97]:


loans["loan_status"].table().head(20)


# In[98]:


ongoing_status = ["Current",
                  "In Grace Period",
                  "Late (16-30 days)",
                  "Late (31-120 days)"]
loans = loans[~loans["loan_status"].isin(ongoing_status)]


# In[99]:


num_filtered_loans = loans.dim[0]
num_filtered_loans


# In[100]:


num_loans_filtered_out = num_unfiltered_loans - num_filtered_loans
num_loans_filtered_out


# ### Create Response Column
# 
# Our response column will be called: `bad_loan`.  The `bad_loan` column will be positive if the loan was not completely paid off.

# In[101]:


loans["bad_loan"] = ~(loans["loan_status"] == "Fully Paid")
loans["bad_loan"] = loans["bad_loan"].asfactor() # convert to enum/factor


# In[102]:


bad_loan_dist = loans["bad_loan"].table()
bad_loan_dist["Percentage"] = (100*bad_loan_dist["Count"]/loans.nrow).round()
bad_loan_dist


# About one in five loans eventually become bad.

# # Step 3 (of 8).  Clean data
# 
# We have multiple columns that are numeric but are being treated as string because of their syntax.  In this section, we will convert these to numeric.  Our machine learning models will have a greater ability to understand numeric features than strings.

# In[103]:


loans[["int_rate", "revol_util", "emp_length"]].head()


# In[104]:


# Convert int_rate to numeric
loans["int_rate"] = loans["int_rate"].gsub(pattern = "%", replacement = "") # strip %
loans["int_rate"] = loans["int_rate"].trim() # trim ws
loans["int_rate"] = loans["int_rate"].asnumeric() #change to a numeric 


# In[105]:


loans["int_rate"].head()


# Now that we have converted interest rate to numeric, we can use the `hist` function to see the distribution of interest rate for good loans and bad loans.

#  The distribution of interest rate is very different for good loans.  This may be a helpful predictor in our model.

# In[106]:


# Convert revol_util to numeric
loans["revol_util"] = loans["revol_util"].gsub(pattern = "%", replacement = "") # strip %
loans["revol_util"] = loans["revol_util"].trim() # trim ws
loans["revol_util"] = loans["revol_util"].asnumeric() #change to a numeric 


# In[107]:


# Convert emp_length to numeric
# Use gsub to remove " year" and " years" also translate n/a to "" 
loans["emp_length"] = loans["emp_length"].gsub(pattern = "([ ]*+[a-zA-Z].*)|(n/a)", replacement = "") 

# Use trim to remove any trailing spaces 
loans["emp_length"] = loans["emp_length"].trim()

# Convert emp_length to numeric 
# Use sub to convert < 1 to 0 years and do the same for 10 + to 10
# Hint: Be mindful of spaces between characters
loans["emp_length"] = loans["emp_length"].gsub(pattern = "< 1", replacement = "0")
loans["emp_length"] = loans["emp_length"].gsub(pattern = "10\\+", replacement = "10")
loans["emp_length"] = loans["emp_length"].asnumeric()


# In[108]:


loans[["int_rate", "revol_util", "emp_length"]].head()


# We can also clean up the verification status column. There are multiple values that mean verified: `VERIFIED - income` and `VERIFIED - income source`.  We will replace these values with `verified`.

# In[109]:


loans["verification_status"].head()


# In[110]:


loans["verification_status"] = loans["verification_status"].sub(pattern = "VERIFIED - income source", 
                                                                replacement = "verified")
loans["verification_status"] = loans["verification_status"].sub(pattern = "VERIFIED - income", 
                                                                replacement = "verified")
loans["verification_status"] = loans["verification_status"].asfactor()


# In[111]:


loans["verification_status"].table()


# # Step 4 (of 8).  Feature engineering
# 
# Now that we have cleaned our data, we can add some new columns to our dataset that may help improve the performance of our supervised learning models.
# 
# The new columns we will create are: 
# * credit_length: the time from their earliest credit line to when they were issued the loan
# * expansion of issue date: extract year and month from the issue date
# * word embeddings from the loan description
# 
# ### Credit Length
# 
# We can extract the credit length by subtracting the year they had their earliest credit line from the year when they issued the loan.

# In[112]:


loans["credit_length"] = loans["issue_d"].year() - loans["earliest_cr_line"].year()
loans["credit_length"].head()


# ### Issue Date Expansion
# 
# We can extract the year and month from the issue date.  We may find that the month or the year when the loan was issued can impact the probability of a bad loan.

# In[113]:


loans["issue_d_year"] = loans["issue_d"].year()
loans["issue_d_month"] = loans["issue_d"].month().asfactor()  # we will treat month as a enum/factor since its cyclical


# In[114]:


loans[["issue_d_year", "issue_d_month"]].head()


# ### Word Embeddings
# 
# One of the columns in our dataset is a description of why the loan was requested. The first few descriptions in the dataset are shown below.

# In[115]:


loans["desc"].head()


# This information may be important to the model but supervised learning algorithms have a hard time understanding text.  Instead we will convert these strings to a numeric vector using the Word2Vec algorithm.

# In[116]:


STOP_WORDS = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what",
              "there","all","we","one","the","a","an","of","or","in","for","by","on",
              "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have",
              "from","at","my","be","by","not","that","to","from","com","org","like","likes","so"]


# In[117]:


def tokenize(sentences, stop_word = STOP_WORDS):
    tokenized = sentences.tokenize("\\W+")
    tokenized_lower = tokenized.tolower()
    tokenized_filtered = tokenized_lower[(tokenized_lower.nchar() >= 2) | (tokenized_lower.isna()),:]
    tokenized_words = tokenized_filtered[tokenized_filtered.grep("[0-9]",invert=True,output_logical=True),:]
    tokenized_words = tokenized_words[(tokenized_words.isna()) | (~ tokenized_words.isin(STOP_WORDS)),:]
    return tokenized_words


# In[118]:


# Break loan description into sequence of words
words = tokenize(loans["desc"].ascharacter())


# In[119]:


# Train Word2Vec Model
from h2o.estimators.word2vec import H2OWord2vecEstimator

w2v_model = H2OWord2vecEstimator(vec_size = 100, model_id = "w2v.hex")
w2v_model.train(training_frame=words)


# In[120]:


# Sanity check - find synonyms for the word 'car'
w2v_model.find_synonyms("car", count = 5)


# In[121]:


# Calculate a vector for each description
desc_vecs = w2v_model.transform(words, aggregate_method = "AVERAGE")


# In[122]:


desc_vecs.head()


# In[123]:


# Add aggregated word embeddings 
loans = loans.cbind(desc_vecs)


# # Step 5 (of 8). Model training
# 
# Now that we have cleaned our data and added new columns, we will train a model to predict bad loans.

# In[124]:


train, test = loans.split_frame(seed = 1234, ratios = [0.75], destination_frames=["train.hex", "test.hex"])


# In[125]:


from h2o.estimators import H2OGradientBoostingEstimator

cols_to_remove = ["initial_list_status",
                  "out_prncp",
                  "out_prncp_inv",
                  "total_pymnt",
                  "total_pymnt_inv",
                  "total_rec_prncp", 
                  "total_rec_int",
                  "total_rec_late_fee",
                  "recoveries",
                  "collection_recovery_fee",
                  "last_pymnt_d", 
                  "last_pymnt_amnt",
                  "next_pymnt_d",
                  "last_credit_pull_d",
                  "collections_12_mths_ex_med" , 
                  "mths_since_last_major_derog",
                  "policy_code",
                  "loan_status",
                  "funded_amnt",
                  "funded_amnt_inv",
                  "mths_since_last_delinq",
                  "mths_since_last_record",
                  "id",
                  "member_id",
                  "desc",
                  "zip_code"]

predictors = list(set(loans.col_names) - set(cols_to_remove))


# In[126]:


predictors


# In[127]:


gbm_model = H2OGradientBoostingEstimator(stopping_metric = "logloss",
                                         stopping_rounds = 5, # early stopping
                                         score_tree_interval = 5,
                                         ntrees = 500,
                                         model_id = "gbm.hex")
gbm_model.train(x = predictors,
                y = "bad_loan",
                training_frame = train,
                validation_frame = test)


# The ROC curve of the training and testing data are shown below.  The area under the ROC curve is much higher for the training data than the testing data indicating that the model may be beginning to memorize the training data.

# In[128]:


print("Training Data")
gbm_model.model_performance(train = True)
print("Testing Data")
gbm_model.model_performance(valid = True)


# # Step 7 (of 8). Interpret model

# In[129]:


loans["inq_last_6mths"].table().head(100)


# # Step 8 (of 8). Stop H2O-3 cluster

# In[133]:


h2o.download_pojo(gbm_model,path="./",get_jar=True)


# In[43]:


h2o.cluster().shutdown()


# # Bonus: Github location for this tutorial
# 
# * https://github.com/h2oai/h2o-tutorials/tree/master/nyc-workshop-2018/h2o_sw/h2o-3-hands-on

# # Bonus: H2O-3 documentation
# 
# * http://docs.h2o.ai
