struct StarField {
    uint32_t star_count;

    float spread;
    float speed;

    float*    star_x;
    float*    star_y;
    float*    star_z;
    uint32_t* star_colour;
};

static void update_and_render_star_field(struct Bitmap* target, struct StarField* field, float delta);
