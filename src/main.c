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
     int32_t* pixel_addrs;
    uint32_t* escaped;
    uint32_t* star_colour;
};

static float rand_zero_to_one() {
    float result = (float)rand() / (float)RAND_MAX + 0.000001f;
    SDL_assert(result >= 0.0f && result <= 1.0f);

    return(result);
}

static uint32_t rand_colour() {
    uint32_t value = (uint32_t)((float)0xFFFFFFFF * ((float)rand() / (float)RAND_MAX));

    return(value);
}

static float ratio = 1.0f / (float)0xFFFFFFFF;
static float t     = 1.0f;

static void init_star(struct StarField* field, size_t star_index, float delta) {
    t += delta / field->phase_speed;

    float zero_to_one = ratio * (float)field->star_colour[star_index];

    field->star_x[star_index] = cos(t);
    field->star_y[star_index] = ((2.0f * zero_to_one) - 1.0f)
        + (
            field->perspective_on_x && field->perspective_on_y
                ? sin(t)
                : 0.0f
        );

    field->star_z[star_index] = rand_zero_to_one() * field->star_spread;
}

static void init_star_field(struct StarField* field) {
    SDL_assert(field->star_count % 4 == 0);

    size_t star_count = field->star_count;
    float* star_pool  = malloc(
          (star_count * sizeof(float) * 3)
        + (star_count * sizeof( int32_t))
        + (star_count * sizeof(uint32_t) * 2)
    );

    field->star_x      = star_pool;
    field->star_y      = star_pool + star_count;
    field->star_z      = star_pool + star_count * 2;

    field->pixel_addrs = ( int32_t*)(star_pool + star_count * 3);
    field->escaped     = (uint32_t*)(star_pool + star_count * 4);
    field->star_colour = (uint32_t*)(star_pool + star_count * 5);

    __m128i mask = _mm_set1_epi32(0x000000FF);

    for (size_t i = 0; i < star_count; i += 4) {
        __m128i colour = _mm_set_epi32(
            rand_colour(),
            rand_colour(),
            rand_colour(),
            rand_colour()
        );

        __m128i value = _mm_or_si128(colour, mask);

        _mm_storeu_si128((__m128i*)(field->star_colour + i), value);
    }

    for (size_t i = 0; i < star_count; i += 1) {
        init_star(field, i, 0.0f);
    }
}

static void deinit_star_field(struct StarField* field) {
    // Star X points to the beginning of the chunk of memory that
    // was allocated for the star data.
    free(field->star_x);
}

static void update_and_render_star_field(struct Bitmap* target, struct StarField* field, float delta) {
    __m128 movement4 = _mm_mul_ps(
        _mm_set1_ps(delta),
        _mm_set1_ps(field->star_speed)
    );

    __m128 zero_epi32     = _mm_set1_epi32(0);
    __m128 half_ps        = _mm_set1_ps(0.5f);
    __m128 width_epi32    = _mm_set1_epi32(target->width);
    __m128 height_epi32   = _mm_set1_epi32(target->height);
    __m128 width_ps       = _mm_set1_ps(target->width);
    __m128 height_ps      = _mm_set1_ps(target->height);
    __m128 half_width_ps  = _mm_mul_ps(width_ps,  half_ps);
    __m128 half_height_ps = _mm_mul_ps(height_ps, half_ps);

    for (size_t i = 0; i < field->star_count; i += 4) {
        float* z_addr = field->star_z + i;

        __m128 x = _mm_loadu_ps(field->star_x + i);
        __m128 y = _mm_loadu_ps(field->star_y + i);
        __m128 z = _mm_loadu_ps(z_addr);

        z = _mm_sub_ps(z, movement4);
        _mm_storeu_ps(z_addr, z);

        __m128i screen_x = _mm_cvtps_epi32(
            _mm_add_ps(
                _mm_mul_ps(
                    field->perspective_on_x
                        ? _mm_div_ps(x, z)
                        : x,
                    half_width_ps
                ),
                half_width_ps
            )
        );

        __m128i screen_y = _mm_cvtps_epi32(
            _mm_add_ps(
                _mm_mul_ps(
                    field->perspective_on_y
                        ? _mm_div_ps(y, z)
                        : y,
                    half_height_ps
                ),
                half_height_ps
            )
        );

        __m128i escaped = zero_epi32;

        // screen_x < 0.0f || screen_x >= width
        escaped = _mm_or_si128(escaped, _mm_cmplt_epi32(screen_x, zero_epi32));
        escaped = _mm_or_si128(escaped, _mm_cmpgt_epi32(screen_x, width_epi32));
        escaped = _mm_or_si128(escaped, _mm_cmpeq_epi32(screen_x, width_epi32));

        // screen_y < 0.0f || screen_y >= width
        escaped = _mm_or_si128(escaped, _mm_cmplt_epi32(screen_y, zero_epi32));
        escaped = _mm_or_si128(escaped, _mm_cmpgt_epi32(screen_y, height_epi32));
        escaped = _mm_or_si128(escaped, _mm_cmpeq_epi32(screen_y, height_epi32));

        // z <= 0.0f
        __m128i z_epi32 = _mm_cvtps_epi32(z);
        escaped = _mm_or_si128(escaped, _mm_cmplt_epi32(z_epi32, zero_epi32));
        escaped = _mm_or_si128(escaped, _mm_cmpeq_epi32(z_epi32, zero_epi32));

        __m128i pixel_addr = _mm_cvtps_epi32(
            _mm_add_ps(_mm_cvtepi32_ps(screen_x), _mm_mul_ps(_mm_cvtepi32_ps(screen_y), width_ps))
        );

        _mm_storeu_si128((__m128i*)(field->escaped + i), escaped);
        _mm_storeu_si128((__m128i*)(field->pixel_addrs + i), pixel_addr);
    }

    for (size_t i = 0; i < field->star_count; i += 1) {
        if (field->escaped[i]) {
            init_star(field, i, delta);
        } else {
            uint32_t* pixel = ((uint32_t*)target->pixels) + field->pixel_addrs[i];
            *pixel = field->star_colour[i];
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
                bitmap_clear(&target);
                update_and_render_star_field(&target, &star_field, delta);
                SDL_UnlockTexture(buffer);

                SDL_RenderCopy(renderer, buffer, NULL, NULL);
                SDL_RenderPresent(renderer);
            }

            deinit_star_field(&star_field);
            SDL_DestroyTexture(buffer);
        }
    }

    return(0);
}
