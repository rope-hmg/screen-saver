
static void bitmap_clear(struct Bitmap* bitmap) {
    memset(bitmap->pixels, 0, bitmap->pitch * bitmap->height);
}
