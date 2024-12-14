use house_price_predictor::{
    download_csv_file, load_csv_file, split_features_and_target, train_test_split,
    train_xgboost_model,
};

//training script and entry point
// Steps
// 1. Download external CSV file to disk
// 2. Load file
// 3. Prepare the data
// 4. Train XGBoost model
// 5. Push to AWS S3 bucket

fn main() -> anyhow::Result<()> {
    println!("Starting training script...");

    // 1. Download external CSV file to disk
    let csv_file_path = download_csv_file()?;

    // 2. load file
    let df = load_csv_file(&csv_file_path)?;

    // 3. Split Data into train and test set
    let (train_df, test_df) = train_test_split(&df, 0.2)?;

    println!("{:?}", train_df.shape());

    // 4. Split into features and target
    let (x_train, y_train) = split_features_and_target(&train_df)?;
    let (x_test, y_test) = split_features_and_target(&test_df)?;

    // 5. Train XGBoost model
    let path_to_model = train_xgboost_model(&x_train, &y_train, &x_test, &y_test)?;

    // 6. Push this model to AWS S3 bucket (model registry)

    Ok(())
}
