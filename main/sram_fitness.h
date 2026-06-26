#ifndef SRAM_FITNESS_H
#define SRAM_FITNESS_H

#include <stdint.h>
#include <stddef.h>
#include "esp_heap_caps.h"

// ---------------------------------------------------------------------------
// Measure-then-decide SRAM fitness
// ---------------------------------------------------------------------------
// The old check (model_flatbuffer_size > 150000) is a poor proxy because:
//   - an int8 model with a small flatbuffer can still demand a large arena,
//   - a model under the threshold can fail AllocateTensors in SRAM anyway,
//   - a model over the threshold might actually fit once quantized.
//
// The only number that predicts "can this run in internal SRAM" is the SUM of
// what we will actually place there:
//     model copy bytes   (memcpy of the flatbuffer into RAM)
//   + arena bytes         (the tensor arena we hand the interpreter)
//   + interpreter/bookkeeping overhead
// ...compared against the largest *contiguous* internal block available right
// now. Internal RAM fragments, so total free heap overstates what you can get
// in one allocation; the largest-free-block is the honest ceiling.
//
// We learn the true arena requirement by constructing the model in PSRAM first
// (where there is headroom) and reading interpreter->arena_used_bytes(). We
// pass arena_used (measured) here, not the arena_size we requested, because the
// requested size is an upper bound you padded for safety, not the real need.
//
// Even after this says "eligible", the SRAM allocation itself is the final
// arbiter: if malloc / AllocateTensors fails, the caller records "didn't fit"
// cleanly instead of crashing. This function only avoids attempts that are
// obviously doomed and gives a human-readable reason for the CSV.
// ---------------------------------------------------------------------------

typedef struct {
    int          eligible;        // 1 => worth attempting an SRAM allocation
    const char*  reason;          // human-readable; goes into the CSV
    size_t       required_bytes;  // model_copy + arena + margin (what we'd place in SRAM)
    size_t       largest_free;    // largest contiguous internal block at decision time
    size_t       total_free;      // total internal free (context only)
} SramFitness;

// Safety margin over the bare requirement. Internal RAM also services FreeRTOS
// task stacks, driver buffers, and interpreter scratch that arena_used_bytes
// does not capture, so we keep a cushion. Tune for your board if needed.
#ifndef SRAM_FITNESS_MARGIN_BYTES
#define SRAM_FITNESS_MARGIN_BYTES (16 * 1024)
#endif

// Fraction of the largest free block we are willing to consume. Leaving slack
// avoids leaving the system with no contiguous block for anything else.
#ifndef SRAM_FITNESS_MAX_FRACTION
#define SRAM_FITNESS_MAX_FRACTION 0.90f
#endif

static inline SramFitness sram_fitness_check(size_t model_copy_bytes,
                                             size_t arena_used_bytes) {
    SramFitness f;
    f.eligible       = 0;
    f.reason         = "";
    f.required_bytes = 0;
    f.largest_free   = 0;
    f.total_free     = 0;

    // Caps: internal (not PSRAM), byte-addressable. This is the heap that the
    // plain malloc()/free() path in your SRAM branch draws from.
    const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

    f.largest_free = heap_caps_get_largest_free_block(caps);
    f.total_free   = heap_caps_get_free_size(caps);

    // What we will actually try to place in SRAM. The model copy and the arena
    // are two separate allocations, but both must come out of the same internal
    // heap, so the sum (plus margin) is the real pressure on it.
    f.required_bytes = model_copy_bytes + arena_used_bytes
                     + (size_t)SRAM_FITNESS_MARGIN_BYTES;

    if (arena_used_bytes == 0) {
        // We never got a valid PSRAM measurement (init failed); refuse to guess.
        f.reason = "no_arena_measurement";
        return f;
    }

    // The arena is the single biggest contiguous allocation we will demand, so
    // it alone must fit in the largest free block. Then the combined footprint
    // must fit within our allowed fraction of that block as a coarse guard
    // against leaving the system starved.
    size_t budget = (size_t)((float)f.largest_free * SRAM_FITNESS_MAX_FRACTION);

    if (arena_used_bytes > f.largest_free) {
        f.reason = "arena_exceeds_largest_block";
        return f;
    }
    if (f.required_bytes > budget) {
        f.reason = "footprint_exceeds_budget";
        return f;
    }

    f.eligible = 1;
    f.reason   = "";
    return f;
}

#endif // SRAM_FITNESS_H