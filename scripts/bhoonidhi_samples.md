# ISRO Bhoonidhi Portal — Sample Data Access Guide

To test SatQuery AI against authentic Indian Space Research Organisation (ISRO) satellite imagery before final evaluation, download free sample packages from the National Remote Sensing Centre (NRSC) Bhoonidhi portal:

### Portal URL
**https://bhoonidhi.nrsc.gov.in/**

---

### Step 1: Register Free Account
1. Visit `https://bhoonidhi.nrsc.gov.in/`
2. Click **User Registration** and register with your student / institutional email.
3. Verify your account.

---

### Step 2: Download Free Sample Products
Under the **Open Data / Sample Products** menu:

1. **Cartosat-2S (High Resolution Optical)**:
   - Search for **Cartosat-2S** sample packages.
   - Download sample **HRMX (High Resolution Multispectral 4-Band)** tiles (2m GSD, VNIR: Blue, Green, Red, NIR).
   - Download sample **PAN (Panchromatic)** tiles (0.65m GSD).

2. **RISAT-1 / EOS-04 (C-band SAR)**:
   - Search for **RISAT-1 / EOS-04** products.
   - Download sample **MRS (Medium Resolution ScanSAR)** or **FRS-1 (Fine Resolution Stripmap)** products.
   - Download sample **Hybrid Polarimetry** packages (containing Stokes parameters S0, S1, S2, S3).

---

### Step 3: Place Samples in Project Directory
Save the unzipped sample GeoTIFFs into:
`d:\SIH\datasets\cartosat_samples\`

---

### Step 4: Run Verification
Run the evaluation test with the sensor normalizer:
```bash
python training/evaluate.py --device cuda
```
The 5-stage sensor normalizer will automatically detect the Cartosat-2S and RISAT metadata tags and verify that the domain gap is cleanly bridged.
