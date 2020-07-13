#include <emmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <xmmintrin.h>

#include <SDL.h>

#define WIDTH  1024
#define HEIGHT 640

struct Bitmap {
    int32_t  pitch;
    int32_t  height;
    uint8_t* pixels;
};

static void bitmap_clear(struct Bitmap* bitmap) {
    memset(bitmap->pixels, 0, bitmap->pitch * bitmap->height);
}

struct StarField {
    uint32_t star_count;

    float spread;
    float speed;

    float*    star_x;
    float*    star_y;
    float*    star_z;
    uint32_t* star_colour;
};

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
    return (
        ((uint32_t)(255.0f * ((float)rand() / (float)RAND_MAX)) << 24) |
        ((uint32_t)(255.0f * ((float)rand() / (float)RAND_MAX)) << 16) |
        ((uint32_t)(255.0f * ((float)rand() / (float)RAND_MAX)) << 8) |
        0xFF
    );
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
    // x, y are in the range -1 to 1.
    field->star_x[star_index] = rand_negative_one_to_one() * field->spread;
    field->star_y[star_index] = rand_negative_one_to_one() * field->spread;

    // z is in the range 0 to 1.
    field->star_z[star_index] = rand_zero_to_one() * field->spread;
}

static void draw_star_field(struct Bitmap* target, struct StarField* field, float delta) {
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

    float half_width  = WIDTH  / 2.0f;
    float half_height = HEIGHT / 2.0f;

    for (size_t i = 0; i < field->star_count; i += 1) {
        float x = field->star_x[i];
        float y = field->star_y[i];
        float z = field->star_z[i];

        if (z <= 0.0f) {
            init_star(field, i);
        }

        int32_t screen_x = (int32_t)((x / z) * half_width  + half_width);
        int32_t screen_y = (int32_t)((y / z) * half_height + half_height);

        if (
            (screen_x < 0 || screen_x >= WIDTH) ||
            (screen_y < 0 || screen_y >= HEIGHT)
        ) {
            init_star(field, i);
        } else {
            uint32_t* pixel = ((uint32_t*)target->pixels) + (screen_x + screen_y * WIDTH);
            *pixel = field->star_colour[i];
        }
    }
}

int main(void) {
    { time_t t; srand(time(&t)); }

    if (SDL_Init(SDL_INIT_VIDEO) == 0) {
        SDL_Window*   window;
        SDL_Renderer* renderer;

        if (SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, SDL_WINDOW_SHOWN, &window, &renderer) == 0) {
            SDL_Texture* buffer = SDL_CreateTexture(
                renderer,
                SDL_PIXELFORMAT_RGBA8888,
                SDL_TEXTUREACCESS_STREAMING,
                WIDTH,
                HEIGHT
            );

            SDL_assert(buffer != NULL);

            struct Bitmap target = {};
            target.height = HEIGHT;

            struct StarField star_field = {};
            star_field.spread     = 64.0f;
            star_field.speed      = 10.0f;
            star_field.star_count = 4096;
            init_star_field(&star_field);

            uint64_t frequency = SDL_GetPerformanceFrequency();
            uint64_t now       = SDL_GetPerformanceCounter();
            uint64_t start;
            double   delta;

            bool is_running = true;
            while (is_running) {
                start = now;
                now   = SDL_GetPerformanceCounter();
                delta = (((double)now - (double)start) * 1000.0) / (double)frequency;

                SDL_Event event;
                while (SDL_PollEvent(&event)) {
                    if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE) {
                        is_running = false;
                    }
                }

                SDL_LockTexture(buffer, NULL, (void**)&target.pixels, &target.pitch);
                draw_star_field(&target, &star_field, delta / 1000.0);
                SDL_UnlockTexture(buffer);

                SDL_RenderCopy(renderer, buffer, NULL, NULL);
                SDL_RenderPresent(renderer);
            }
        }
    }

    return(0);
}
