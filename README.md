CTH to CTA/CTP Image Translation using Deep Learning

Overview:

  This repository contains code to generate Time-to-Maximum (Tmax) and Cerebral Blood Flow (CBF) maps from Non-Contrast CT Head (NCCTH) images using a deep learning model. This work aims to expedite triage and decision-making for patients with acute ischemic stroke (AIS) due to large-vessel occlusions (LVO) by providing an alternative to CT perfusion (CTP) imaging, which can cause delays in treatment. The project uses the pix2pix-turbo model, a conditional generative adversarial network based on Stable Diffusion Turbo, for image-to-image translation from NCCTH to perfusion maps.

Purpose:

  Endovascular thrombectomy (EVT) is the standard of care for treating AIS caused by LVO. Although CTP imaging is often used to select candidates for EVT, the time taken for CTP acquisition and processing can delay critical treatment, potentially worsening patient outcomes. This project leverages generative deep learning techniques to generate key perfusion maps—Tmax and CBF—directly from NCCTH images, aiming to bypass the need for separate CTP imaging.

Methods and Materials:

  Dataset
  Patients presenting with AIS due to LVO were retrospectively reviewed from a multi-site academic medical center between May 2019 and June 2021.
  Included patients had both NCCTH and CTP images acquired within a 30-minute interval.
  Tmax and CBF maps were generated from CTP images using Viz.ai post-processing software and were used as ground truth labels.
  Brain structures were segmented from NCCTH using the TotalSegmentator package.
  NCCTH slices were co-registered to the corresponding CTP images.
  Model
  
  Model Type: pix2pix-turbo, a conditional generative adversarial network based on Stable Diffusion Turbo.
  Task: Paired image-to-image translation, predicting Tmax and CBF maps from co-registered NCCTH slices.
  
  Training
  101 patients were included after applying exclusion criteria, with a split of 90% for training and 10% for testing.
  The model was trained to predict:
  
  Results
  Tmax Maps: Fréchet Inception Distance (FID): 62.21, Peak Signal-to-Noise Ratio (PSNR): 16.99, Structural Similarity Index (SSIM): 0.827.
  CBF Maps: FID: 60.10, PSNR: 16.38, SSIM: 0.702.
  The model demonstrated moderate efficacy in generating Tmax and CBF maps from NCCTH images, as indicated by FID, PSNR, and SSIM metrics.
  These results indicate the potential of deep learning approaches for cross-modality translation from NCCTH to perfusion imaging.

Conclusion and Future Work:
  The developed generative AI model demonstrated its ability to directly produce Tmax and CBF maps from NCCTH images, which could lead to faster, more informed triage and treatment decisions in acute stroke scenarios. Future work will focus on:
  Validating the findings with larger datasets.
  Predicting estimates of Tmax and CBF volumes.
