use polars::prelude::*;
use rand::{seq::SliceRandom, thread_rng};
use xgboost::{parameters, Booster, DMatrix, IndexOrder};

pub fn download_csv_file() -> anyhow::Result<String> {
    println!("Downloading CSV file...");

    // get response from url
    let url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";

    // get the response from the URL
    let response = reqwest::blocking::get(url)?; // question mark basically raises an error if the result is an error

    // Copy these bytes to a file on disk
    let bytes = response.bytes()?;

    let file_path = "boston_housing.csv";

    // Copy the bytes to a file on disk
    std::fs::write(file_path, bytes)?;

    Ok(file_path.to_string())
}

pub fn load_csv_file(file_path: &str) -> anyhow::Result<DataFrame> {
    let df = CsvReader::from_path(file_path)?.finish()?;

    println!("Loaded {} rows and {} columns", df.height(), df.width());
    println!("{:?}", df.head(Some(5)));

    Ok(df)
}

pub fn train_test_split(
    df: &DataFrame,
    perc_test_size: f64,
) -> anyhow::Result<(DataFrame, DataFrame)> {
    // generate vector from 1 to # of rows in df
    let mut indices: Vec<usize> = (0..df.height()).collect();

    // create rng
    let mut rng = thread_rng();

    // shuffle indices
    indices.shuffle(&mut rng);

    // split the indices into training and testing
    let split_idx = (df.height() as f64 * perc_test_size) as usize;

    // Create the training and testing sets
    let train_indices = indices[0..split_idx].to_vec();
    let test_indices = indices[split_idx..].to_vec();

    // convert Vec<i64> to chunked ChunekedArray<Int64Type>
    let train_indices_ca =
        UInt32Chunked::from_vec("", train_indices.iter().map(|&x| x as u32).collect());
    let test_indices_ca =
        UInt32Chunked::from_vec("", test_indices.iter().map(|&x| x as u32).collect());

    // Create the training and testing DataFrames
    let train_df = df.take(&train_indices_ca)?;
    let test_df = df.take(&test_indices_ca)?;

    Ok((train_df, test_df))
}

pub fn split_features_and_target(df: &DataFrame) -> anyhow::Result<(DataFrame, DataFrame)> {
    let feature_names = vec![
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b",
        "lstat",
    ];

    let target_name = vec!["medv"];

    let features = df.select(feature_names)?;
    let target = df.select(target_name)?;

    Ok((features, target))
}

// Train xgboost model
// Evaluates performance with test data
// Saves model locally and returns path to the generated model file
pub fn train_xgboost_model(
    x_train: &DataFrame,
    y_train: &DataFrame,
    x_test: &DataFrame,
    y_test: &DataFrame,
) -> anyhow::Result<String> {
    // Transform Polars Dataframe into 2D array in row-major order
    let x_train_array = x_train.to_ndarray::<Float32Type>(IndexOrder::C);
    let y_train_array = y_train.to_ndarray::<Float32Type>(IndexOrder::C);
    let x_test_array = x_test.to_ndarray::<Float32Type>(IndexOrder::C);
    let y_test_array = y_test.to_ndarray::<Float32Type>(IndexOrder::C);

    let x_train_slice = x_train_array.as_slice().unwrap();
    let y_train_slice = y_train_array.as_slice().unwrap();
    let x_test_slice = x_test_array.as_slice().unwrap();
    let y_test_slice = y_test_array.as_slice().unwrap();

    // transform given DataFrames into XGBoost DMatrix objects
    let mut dmatrix_train =
        DMatrix::from_dense(x_train_slice, x_train.height())?.set_labels(y_train_slice);

    let mut dmatrix_test =
        DMatrix::from_dense(x_test_slice, x_test.height())?.set_labels(y_test_slice);

    let evaluation_sets = &[(&dmatrix_terain, "train"), (&dmatrix_test, "test")];

    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dmatrix_train)
        .evaluation_sets(Some(evaluation_sets))
        // .custom_objective_fn(parameters::::RegSquareError)
        // .custom_evaluation_fn(parameters::Objective::)
        .build()
        .unwrap();

    let model = Booster::train(&training_params).unwrap();

    // Evaluate the model on the test set
    // TODO: Investigate what error metrics to use for regression

    println!("Test {:?}", model.predict(&dmatrix_test).unwrap());

    // Save the model to a file
    let model_path = "boston_housing_model.bin";
    model.save(model_path)?;

    Ok(model_path.to_string())
}
