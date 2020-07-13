struct Bitmap {
    int32_t  width;
    int32_t  height;
    int32_t  pitch;
    uint8_t* pixels;
};

static void bitmap_clear(struct Bitmap* bitmap);
