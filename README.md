# pneumonia_detection_analysis

## Database Set Up
We will be hosting our data on AWS through the use of the S3 Buckets and the a postgreSQL RDS. Our dataset has over 5,000 images of chest x-rays that will be run through our machine learning model to determine if we can predict whether or not someone has Pneumonia. We chose to use AWS since it can easily store non-text data (images), our data is stored in the cloud so everyone can access it from their local devices, and we can upload our final data into a RDS for future querying and analysis. 

### S3 Bucket Links
- Test: s3://pneumonia-detection-analysis/test/
    - Normal: s3://pneumonia-detection-analysis/test/NORMAL/
    - Pneumonia: s3://pneumonia-detection-analysis/test/PNEUMONIA/

- Train: s3://pneumonia-detection-analysis/train/
    - Normal: s3://pneumonia-detection-analysis/train/NORMAL/
    - Pneumonia: s3://pneumonia-detection-analysis/train/PNEUMONIA/

- Val: s3://pneumonia-detection-analysis/val/
    - Normal: s3://pneumonia-detection-analysis/val/NORMAL/
    - Pneumonia: s3://pneumonia-detection-analysis/val/PNEUMONIA/

### RDS Endpoint
- pneumonia-detection-analysis.cyhi4xykqawo.us-east-1.rds.amazonaws.com
