# Local simulation raster inputs

Place the fixed Milan Layer Alterator inputs here:

```text
simulation_inputs/
├── ucps/
│   ├── TCH.tif
│   ├── IMD.tif
│   ├── BH.tif
│   ├── BSF.tif
│   └── SVF.tif
└── lc_fractions/
    ├── F_AC.tif
    ├── F_S.tif
    ├── F_M.tif
    ├── F_BS.tif
    ├── F_G.tif
    ├── F_TV.tif
    └── F_W.tif
```

The web application discovers these folders automatically. GeoTIFF files are
local study-area data and are intentionally excluded from Git because they are
approximately 440 MB in total.
