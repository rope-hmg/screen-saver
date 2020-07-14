#include <emmintrin.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xmmintrin.h>

#include <SDL.h>

#define PI 3.14159265358979323846

#define WIDTH  1024
#define HEIGHT 640

struct Bitmap {
    int32_t  width;
    int32_t  height;
    int32_t  pitch;
    uint8_t* pixels;
};

static void bitmap_clear(struct Bitmap* bitmap) {
    memset(bitmap->pixels, 0, bitmap->pitch * bitmap->height);
}

struct StarField {
    bool     perspective_on_x;
    bool     perspective_on_y;
    uint32_t star_count;
    float    star_speed;
    float    star_spread;
    float    phase_speed;

    float*    star_x;
    float*    star_y;
    float*    star_z;
    uint32_t* star_colour;
};

static float rand_zero_to_one() {
    float result = (float)rand() / (float)RAND_MAX + 0.000001f;
    SDL_assert(result >= 0.0f && result <= 1.0f);
    return(result);
}

static uint32_t rand_colour() {
    uint32_t value = (uint32_t)((float)0xFFFFFFFF * ((float)rand() / (float)RAND_MAX));
    uint32_t mask  = 0x000000FF;
    return(value | mask);
}

static float ratio = 1.0f / (float)0xFFFFFFFF;
static float t     = 0.0f;

static void init_star(struct StarField* field, size_t star_index, float delta) {
    t += delta / field->phase_speed;

    float zero_to_one = ratio * (float)field->star_colour[star_index];
    float negative_one_to_one = (2.0f * zero_to_one) - 1.0f;

    field->star_x[star_index] = cos(t);
    field->star_y[star_index] = field->perspective_on_x && field->perspective_on_y
        ? negative_one_to_one + sin(t)
        : negative_one_to_one;
    field->star_z[star_index] = rand_zero_to_one() * field->star_spread;
}

static void init_star_field(struct StarField* field) {
    SDL_assert(field->star_count % 4 == 0);

    size_t star_count = field->star_count;
    float* star_pool  = malloc(star_count * sizeof(float) * 3);

    field->star_x = star_pool;
    field->star_y = star_pool + star_count;
    field->star_z = star_pool + star_count * 2;

    field->star_colour = malloc(star_count * sizeof(uint32_t));

    for (size_t i = 0; i < star_count; i += 1) {
        field->star_colour[i] = rand_colour();
        init_star(field, i, 0.0f);
    }
}

static void update_and_render_star_field(struct Bitmap* target, struct StarField* field, float delta) {
    bitmap_clear(target);

    __m128 delta4 = _mm_set_ps1(delta);
    __m128 speed4 = _mm_set_ps1(field->star_speed);

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
            init_star(field, i, delta);
        } else {
            float x_perspective = field->perspective_on_x ? z : 1.0f;
            float y_perspective = field->perspective_on_y ? z : 1.0f;

            int32_t screen_x = (int32_t)((x / x_perspective) * half_width  + half_width);
            int32_t screen_y = (int32_t)((y / y_perspective) * half_height + half_height);

            if (
                (screen_x < 0 || screen_x >= target->width) ||
                (screen_y < 0 || screen_y >= target->height)
            ) {
                init_star(field, i, delta);
            } else {
                uint32_t* pixel = ((uint32_t*)target->pixels) + (screen_x + screen_y * target->width);
                *pixel = field->star_colour[i];
            }
        }
    }
}

int main(void) {
    { time_t t; srand((unsigned) time(&t)); }

    if (SDL_Init(SDL_INIT_VIDEO) == 0) {
        SDL_Window*   window;
        SDL_Renderer* renderer;

        if (SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, SDL_WINDOW_SHOWN, &window, &renderer) == 0) {
            // If we fail to get the display mode information or the refresh rate is
            // unspecified or for what ever reason, we will default to 60Hz.
            int32_t refresh_rate  = 60;
            int32_t display_index = SDL_GetWindowDisplayIndex(window);

            SDL_DisplayMode display_mode;
            if (SDL_GetDesktopDisplayMode(display_index, &display_mode) == 0) {
                refresh_rate = (display_mode.refresh_rate == 0)
                    ? refresh_rate
                    : display_mode.refresh_rate;
            }

            double target_seconds_per_frame = 1.0 / (double)refresh_rate;

            SDL_Texture* buffer = SDL_CreateTexture(
                renderer,
                SDL_PIXELFORMAT_RGBA8888,
                SDL_TEXTUREACCESS_STREAMING,
                WIDTH,
                HEIGHT
            );

            SDL_assert(buffer != NULL);

            struct Bitmap target = {};
            target.width  = WIDTH;
            target.height = HEIGHT;

            struct StarField star_field = {};
            star_field.perspective_on_x = true;
            star_field.perspective_on_y = true;
            star_field.star_count       = 16384;
            star_field.star_speed       = 5.0f;
            star_field.star_spread      = 16.0f;
            star_field.phase_speed      = 64.0f;

            init_star_field(&star_field);

            uint64_t frequency = SDL_GetPerformanceFrequency();
            uint64_t now       = SDL_GetPerformanceCounter();
            uint64_t start;
            double   delta;

            bool is_running = true;
            while (is_running) {
                start = now;
                now   = SDL_GetPerformanceCounter();
                delta = ((double)now - (double)start) / (double)frequency;

                // NOTE(Hector):
                // The timing stuff calculated above is for the previous frame, so we're going to wait
                // here to make sure that we're actually taking the correct amount of time per frame.
                // I'm not 100% sure we need to do this, since we're already using vsync. I guess just
                // because my monitor allows vsync, doesn't mean they all do. The quandaries...
                if (delta < target_seconds_per_frame) {
                    uint32_t ms_to_sleep = (uint32_t)(target_seconds_per_frame - delta) * 1000;
                    if (ms_to_sleep >= 1) ms_to_sleep -= 1;

                    SDL_Delay(ms_to_sleep);

                    #define SecondsElapsed() ((double)SDL_GetPerformanceCounter() - (double)start) / (double)frequency
                    while (SecondsElapsed() < target_seconds_per_frame);
                }

                SDL_Event event;
                while (SDL_PollEvent(&event)) {
                    if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE) {
                        is_running = false;
                    }
                }

                SDL_LockTexture(buffer, NULL, (void**)&target.pixels, &target.pitch);
                update_and_render_star_field(&target, &star_field, delta);
                SDL_UnlockTexture(buffer);

                SDL_RenderCopy(renderer, buffer, NULL, NULL);
                SDL_RenderPresent(renderer);
            }
        }
    }

    return(0);
}
