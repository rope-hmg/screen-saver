static float rand_negative_one_to_one() {
    float result = 2.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
    SDL_assert(result >= -1.0f && result <= 1.0f);
    return(result);
}

static float rand_zero_to_one() {
    float result = (float)rand() / (float)RAND_MAX + 0.000001f;
    SDL_assert(result >= 0.0f && result <= 1.0f);
    return(result);
}

static uint32_t rand_colour() {
    uint32_t value = (uint32_t)((float)0xFFFFFFFF * ((float)rand() / (float)RAND_MAX));
    uint32_t mask  = 0x000000FF;
    return (value | mask);
}

static void init_star_field(struct StarField* field) {
    SDL_assert(field->star_count % 4 == 0);

    size_t star_count = field->star_count;
    float* star_pool  = malloc(star_count * sizeof(float) * 3);

    field->star_x = star_pool;
    field->star_y = star_pool + star_count;
    field->star_z = star_pool + star_count * 2;

    field->star_colour = malloc(star_count * sizeof(uint32_t));

    __m128 spread = _mm_set_ps1(field->spread);

    for (size_t i = 0; i < star_count; i += 4) {
        __m128 rand_x = _mm_set_ps(
            rand_negative_one_to_one(),
            rand_negative_one_to_one(),
            rand_negative_one_to_one(),
            rand_negative_one_to_one()
        );

        _mm_store_ps(field->star_x + i, _mm_mul_ps(rand_x, spread));
    }

    for (size_t i = 0; i < star_count; i += 4) {
        __m128 rand_y = _mm_set_ps(
            rand_negative_one_to_one(),
            rand_negative_one_to_one(),
            rand_negative_one_to_one(),
            rand_negative_one_to_one()
        );

        _mm_store_ps(field->star_y + i, _mm_mul_ps(rand_y, spread));
    }

    for (size_t i = 0; i < star_count; i += 4) {
        __m128 rand_z = _mm_set_ps(
            rand_zero_to_one(),
            rand_zero_to_one(),
            rand_zero_to_one(),
            rand_zero_to_one()
        );

        _mm_store_ps(field->star_z + i, _mm_mul_ps(rand_z, spread));
    }

    for (size_t i = 0; i < star_count; i += 4) {
        _mm_store_si128(
            (__m128i*)(field->star_colour + i),
            _mm_set_epi32(
                rand_colour(),
                rand_colour(),
                rand_colour(),
                rand_colour()
            )
        );
    }
}

static void init_star(struct StarField* field, size_t star_index) {
    float r = rand_zero_to_one();

    if (r > 0.2f) {
        // x, y are in the range -1 to 1.
        field->star_x[star_index] = rand_negative_one_to_one() * field->spread;
        field->star_y[star_index] = rand_negative_one_to_one() * field->spread;
    } else {
        static float ratio = 1.0f / (float)0xFFFFFFFF;
        float zero_to_one = ratio * (float)field->star_colour[star_index];
        float negative_one_to_one = (2.0f * zero_to_one) - 1.0f;

        field->star_x[star_index] = 1.0f / negative_one_to_one;
        field->star_y[star_index] = negative_one_to_one;
    }

    // z is in the range 0 to 1.
    field->star_z[star_index] = rand_zero_to_one() * field->spread;
}

static void update_and_render_star_field(struct Bitmap* target, struct StarField* field, float delta) {
    bitmap_clear(target);

    __m128 delta4 = _mm_set_ps1(delta);
    __m128 speed4 = _mm_set_ps1(field->speed);

    for (size_t i = 0; i < field->star_count; i += 4) {
        float* addr = field->star_z + i;

        _mm_store_ps(
            addr,
            _mm_sub_ps(
                _mm_load_ps(addr),
                _mm_mul_ps(delta4, speed4)
            )
        );
    }

    float half_width  = target->width  * 0.5f;
    float half_height = target->height * 0.5f;

    for (size_t i = 0; i < field->star_count; i += 1) {
        float x = field->star_x[i];
        float y = field->star_y[i];
        float z = field->star_z[i];

        if (z <= 0.0f) {
            init_star(field, i);
        } else {
            int32_t screen_x = (int32_t)((x / z) * half_width  + half_width);
            int32_t screen_y = (int32_t)((y / z) * half_height + half_height);

            if (
                (screen_x < 0 || screen_x >= target->width) ||
                (screen_y < 0 || screen_y >= target->height)
            ) {
                init_star(field, i);
            } else {
                uint32_t* pixel = ((uint32_t*)target->pixels) + (screen_x + screen_y * target->width);
                *pixel = field->star_colour[i];
            }
        }
    }
}
