CREATE TABLE train_data (
 imgID INT PRIMARY KEY NOT NULL,
 imgName TEXT,
 IsPneumonia boolean,
 IsBacteria boolean,
 IsVirus boolean
);
CREATE TABLE test_data (
 imgID INT PRIMARY KEY NOT NULL,
 imgName TEXT,
 IsPneumonia boolean,
 IsBacteria boolean,
 IsVirus boolean,
 Predicted_output boolean
);