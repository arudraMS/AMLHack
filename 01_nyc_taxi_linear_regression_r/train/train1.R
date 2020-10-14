#' Copyright(c) Microsoft Corporation.
#' Licensed under the MIT license.

library(azuremlsdk)
library(optparse)
#library(caret)

options <- list(
  make_option(c("-d", "--data_folder")),
  make_option(c("-u", "--username"))
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

#printing out to log the data folder
paste(opt$data_folder)

#printing out to the log the user name
paste(opt$username)



#ws = run.experiment.workspace
#paste(ws)

#current_run <- get_current_run(allow_offline = TRUE)


#option 1 - read from file
filename <- paste(opt$username, "green-taxi.Rd", sep="-")
paste(filename)

paste("about to read RDS")
df <- readRDS(file.path(opt$data_folder, filename))

#filename = ".\train\memasanz-green-taxi.Rd"
#df <- readRDS(filename)


#option 2 - get from public location
#print("About to summarize the dataset")
#dataset <- create_tabular_dataset_from_parquet_files(path="https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/green_taxi_data.parquet",
#                                                     validate = TRUE,
#                                                     include_path = FALSE,
#                                                     set_column_types = NULL,
#                                                     partition_format = NULL
#)
#
#df <- load_dataset_into_data_frame(dataset,on_error = "null",out_of_range_datetime = "null")

##option 3 - 
#ws <- load_workspace_from_config()
#ds <- get_default_datastore(ws)
#
#target_path <- paste(username, "greentaxi", sep="-")
#df <- download_from_datastore(ds,
#                          list(paste("./", filename, sep="")),
#                          target_path = target_path,
#                          overwrite = TRUE)


summary(df)
#-----------------------------
set.seed(42)

#------------------------
#df_idx <- createDataPartition(df$totalAmount, p = 0.80, list = FALSE)
sample_size_split <- as.integer(nrow(df) * .80)

print(nrow(df))
print(sample_size_split)

df_idx <- sample(1:nrow(df), sample_size_split)
df_train <- df[df_idx, ]
nrow(df_train)
df_test <- df[-df_idx, ]
nrow(df_test)

RMSE = function(model){
  RMSE_train = sqrt(mean(resid(model) ^ 2))
  RMSE_test = sqrt(mean(( df_test$totalAmount - predict(model, newdata = df_test))^2))
  
  values = c(
    RMSE_train = RMSE_train, 
    RMSE_test = RMSE_test)
}


#------------------------------------------

print("About to generate model Model 1")
mod = lm(totalAmount ~ tripDistance , data = df_train)


print("Summary of Model")
summary(mod)

results = RMSE(mod)

log_metric_to_run("R^2", summary(mod)$r.squared)
log_metric_to_run("RMSE Train", results['RMSE_train'])
log_metric_to_run("RMSE Test", results['RMSE_test'])


summary(mod)$r.squared

summary(mod)$coef


output_dir = "outputs"
if (!dir.exists(output_dir)){
  dir.create(output_dir)
}
saveRDS(mod, file = "./outputs/model.rds")
message("Model saved")