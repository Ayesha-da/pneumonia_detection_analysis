# pneumonia_detection_analysis

## Database Set Up
We will be hosting our data on AWS through the use of the S3 Buckets and the a postgreSQL RDS. Our dataset has over 5,000 images of chest x-rays that will be run through our machine learning model to determine if we can predict whether or not someone has Pneumonia. We chose to use AWS since it can easily store non-text data (images), our data is stored in the cloud so everyone can access it from their local devices, and we can upload our final data into a RDS for future querying and analysis. 

### S3 Bucket Links
- Test: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=test/&showversions=false
    - Normal: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=test/NORMAL/&showversions=false
    - Pneumonia: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=test/PNEUMONIA/&showversions=false

- Train: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=train/&showversions=false
    - Normal: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=train/NORMAL/&showversions=false
    - Pneumonia: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=train/PNEUMONIA/&showversions=false

- Val: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=val/&showversions=false
    - Normal: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=val/NORMAL/&showversions=false
    - Pneumonia: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=val/PNEUMONIA/&showversions=false

### RDS Endpoint
- pneumonia-detection-analysis.cyhi4xykqawo.us-east-1.rds.amazonaws.com
