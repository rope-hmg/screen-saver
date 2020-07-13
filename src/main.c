#include <emmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <xmmintrin.h>

#include <SDL.h>

#include "bitmap.h"
#include "star_field.h"

#include "bitmap.c"
#include "star_field.c"

#define WIDTH  1024
#define HEIGHT 640

int main(void) {
    { time_t t; srand((unsigned) time(&t)); }

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
            target.width  = WIDTH;
            target.height = HEIGHT;

            struct StarField star_field = {};
            star_field.spread     = 64.0f;
            star_field.speed      = 10.0f;
            star_field.star_count = 16384;

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
                update_and_render_star_field(&target, &star_field, delta / 1000.0);
                SDL_UnlockTexture(buffer);

                SDL_RenderCopy(renderer, buffer, NULL, NULL);
                SDL_RenderPresent(renderer);
            }
        }
    }

    return(0);
}
