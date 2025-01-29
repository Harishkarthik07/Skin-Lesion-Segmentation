CREATE DATABASE skin_lesion_segmentation;
USE skin_lesion_segmentation;

CREATE TABLE uploads (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mask_path VARCHAR(255)
);
