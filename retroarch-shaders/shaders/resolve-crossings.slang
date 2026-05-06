#version 450
#pragma format R32G32B32A32_SFLOAT

// Pass 1: Resolve diagonal crossings in the similarity graph.
//
// Each output pixel corresponds to a graph position. Only corner positions
// (even,even) with crossing flags == 3 (both diagonals) need resolution.
// All other positions are passed through unchanged.
//
// Three heuristics vote to decide which diagonal to keep:
//   1. Curve length — longer connected chain wins (walk_chain)
//   2. Sparse pixels — smaller component in 8x8 BFS window wins
//   3. Islands — valence-1 endpoint gets fixed weight 5
//
// Fully-connected blocks (all 4 cardinal edges) remove both diagonals.
// Single diagonals in fully-connected blocks are also removed.
// Ties remove both diagonals (matches reference implementation).
//
// Reads from Source (similarity graph, Pass 0 output).

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
} params;

layout(std140, set = 0, binding = 0) uniform UBO {
    mat4 MVP;
} global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;

void main() {
    gl_Position = global.MVP * Position;
    vTexCoord = TexCoord;
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;

layout(set = 0, binding = 2) uniform sampler2D Source; // Pass 0 output (similarity graph)

// ---- Graph access helpers ----

// Image dimensions (original input, not the 2x graph)
// SourceSize.xy = 2*imgW, 2*imgH (the graph texture size)
// OriginalSize.xy = imgW, imgH (the original pixel art)

uint img_w() { return uint(params.OriginalSize.x); }
uint img_h() { return uint(params.OriginalSize.y); }

// Read graph edge/diagonal flag at graph coordinate (gx, gy).
// The graph texture is 2W x 2H; graph coords 0..2W-1, 0..2H-1 are valid.
// Values beyond that are border (0).
//
// Pixel colors at odd,odd are stored as normalized RGB in .rgb (not used here).
// Edge connectivity (even,odd / odd,even) and diagonal flags (even,even) are
// small integers stored as float(value) in .r. Read back with uint(val + 0.5).
// This function is only called for edge/diagonal positions (never odd,odd).
uint g(int gx, int gy) {
    ivec2 texSize = ivec2(params.SourceSize.xy);
    if (gx < 0 || gy < 0 || gx >= texSize.x || gy >= texSize.y)
        return 0u;
    vec4 val = texelFetch(Source, ivec2(gx, gy), 0);
    return uint(val.r + 0.5);
}

// ---- Direction constants (matching compute shader) ----

const uint DIR_NW = 1u;
const uint DIR_W  = 2u;
const uint DIR_SW = 4u;
const uint DIR_S  = 8u;
const uint DIR_SE = 16u;
const uint DIR_E  = 32u;
const uint DIR_NE = 64u;
const uint DIR_N  = 128u;

// ---- Valence computation ----

// Compute 8-connected valence bitmask for pixel (x,y).
// Cardinal edges are at (2x+1 +/- 1, 2y+1) and (2x+1, 2y+1 +/- 1).
// Diagonal edges are corner flags at adjacent corners.
uint valence_mask(uint x, uint y) {
    uint v = 0u;
    int gx = int(2u * x + 1u);
    int gy = int(2u * y + 1u);
    int gw = int(params.SourceSize.x);
    int gh = int(params.SourceSize.y);

    // Cardinal
    if (gx >= 2 && g(gx - 1, gy) != 0u) v |= DIR_W;
    if (gx + 1 < gw && g(gx + 1, gy) != 0u) v |= DIR_E;
    if (gy >= 2 && g(gx, gy - 1) != 0u) v |= DIR_N;
    if (gy + 1 < gh && g(gx, gy + 1) != 0u) v |= DIR_S;

    // Diagonal: check corner flags
    // NW corner is at (gx-1, gy-1), backslash bit 0 means TL-BR, we are BR = SE endpoint
    if (gx >= 2 && gy >= 2 && (g(gx - 1, gy - 1) & 1u) != 0u) v |= DIR_NW;
    // NE corner is at (gx+1, gy-1), slash bit 1 means TR-BL, we are BL = SW endpoint
    if (gx + 1 < gw && gy >= 2 && (g(gx + 1, gy - 1) & 2u) != 0u) v |= DIR_NE;
    // SW corner is at (gx-1, gy+1), slash bit 1
    if (gx >= 2 && gy + 1 < gh && (g(gx - 1, gy + 1) & 2u) != 0u) v |= DIR_SW;
    // SE corner is at (gx+1, gy+1), backslash bit 0
    if (gx + 1 < gw && gy + 1 < gh && (g(gx + 1, gy + 1) & 1u) != 0u) v |= DIR_SE;

    return v;
}

uint valence_count(uint mask) {
    return uint(bitCount(mask));
}

// ---- Chain walking ----

// Walk a chain of valence-2 nodes using XOR direction tracking.
uint walk_chain(uint sx, uint sy, uint fx, uint fy) {
    uint cx = sx, cy = sy;
    uint length = 1u;

    int idx = int(fx) - int(sx);
    int idy = int(fy) - int(sy);
    uint pred_dir = 0u;
    if (idx ==  1 && idy ==  0) pred_dir = DIR_E;
    if (idx == -1 && idy ==  0) pred_dir = DIR_W;
    if (idx ==  0 && idy ==  1) pred_dir = DIR_S;
    if (idx ==  0 && idy == -1) pred_dir = DIR_N;
    if (idx ==  1 && idy ==  1) pred_dir = DIR_SE;
    if (idx == -1 && idy == -1) pred_dir = DIR_NW;
    if (idx ==  1 && idy == -1) pred_dir = DIR_NE;
    if (idx == -1 && idy ==  1) pred_dir = DIR_SW;

    for (uint iter = 0u; iter < 200u; iter++) {
        uint mask = valence_mask(cx, cy);
        if (valence_count(mask) != 2u) break;

        uint next_dir = mask ^ pred_dir;

        int dx = 0, dy = 0;
        uint reverse = 0u;
        if      (next_dir == DIR_N)  { dy = -1; reverse = DIR_S; }
        else if (next_dir == DIR_NE) { dx =  1; dy = -1; reverse = DIR_SW; }
        else if (next_dir == DIR_E)  { dx =  1; reverse = DIR_W; }
        else if (next_dir == DIR_SE) { dx =  1; dy =  1; reverse = DIR_NW; }
        else if (next_dir == DIR_S)  { dy =  1; reverse = DIR_N; }
        else if (next_dir == DIR_SW) { dx = -1; dy =  1; reverse = DIR_NE; }
        else if (next_dir == DIR_W)  { dx = -1; reverse = DIR_E; }
        else if (next_dir == DIR_NW) { dx = -1; dy = -1; reverse = DIR_SE; }
        else { break; } // not a single direction bit

        cx = uint(int(cx) + dx);
        cy = uint(int(cy) + dy);
        pred_dir = reverse;
        length++;
    }
    return length;
}

// ---- Sparse pixel BFS ----

// Pack 64 labels (each 0/1/2, 2 bits) into uvec4 (128 bits). Keeps the
// labels array in registers instead of spilling 256 bytes of int[64] to
// scratch. idx = row * 8 + col; uint_idx = idx / 16; shift = (idx % 16) * 2.
uint get_label(uvec4 labels, int row, int col) {
    int idx = row * 8 + col;
    int uint_idx = idx >> 4;
    int shift = (idx & 15) << 1;
    uint v;
    if (uint_idx == 0)      v = labels.x;
    else if (uint_idx == 1) v = labels.y;
    else if (uint_idx == 2) v = labels.z;
    else                    v = labels.w;
    return (v >> shift) & 3u;
}

void set_label(inout uvec4 labels, int row, int col, uint value) {
    int idx = row * 8 + col;
    int uint_idx = idx >> 4;
    int shift = (idx & 15) << 1;
    uint clear = ~(3u << shift);
    uint v = (value & 3u) << shift;
    if (uint_idx == 0)      labels.x = (labels.x & clear) | v;
    else if (uint_idx == 1) labels.y = (labels.y & clear) | v;
    else if (uint_idx == 2) labels.z = (labels.z & clear) | v;
    else                    labels.w = (labels.w & clear) | v;
}

// Simultaneous A/B component labeling in 8x8 window around crossing.
// Center 2x2 seeded with A(1) and B(2), then multi-source BFS propagates.
// Returns component sizes excluding seeds.
void sparse_pixel_sizes(uint bx, uint by, out uint sizeA, out uint sizeB) {
    // 8x8 labels packed into uvec4. Seeds at (col,row) ∈ {(3,3),(4,4)}=1
    // (main/backslash) and {(4,3),(3,4)}=2 (anti/slash):
    //   idx 27 (row 3, col 3) → uint_idx 1, shift 22 → 1u << 22 in .y
    //   idx 28 (row 3, col 4) → uint_idx 1, shift 24 → 2u << 24 in .y
    //   idx 35 (row 4, col 3) → uint_idx 2, shift  6 → 2u <<  6 in .z
    //   idx 36 (row 4, col 4) → uint_idx 2, shift  8 → 1u <<  8 in .z
    uvec4 labels = uvec4(
        0u,
        (1u << 22) | (2u << 24),
        (2u <<  6) | (1u <<  8),
        0u
    );

    sizeA = 0u;
    sizeB = 0u;

    // BFS frontier: bit i set = cell idx i (= row*8+col) needs processing.
    // 64 bits = uvec2, lives in registers. Eliminates the BFS queue scratch
    // entirely. Each cell is added to the frontier exactly once (when first
    // labeled), and popped exactly once via findLSB. No work is duplicated.
    //   bit 27 (row 3, col 3) and bit 28 (row 3, col 4) → frontier.x
    //   bit 35 (row 4, col 3) and bit 36 (row 4, col 4) → frontier.y at
    //                                                     bit 3 / 4.
    uvec2 frontier = uvec2(
        (1u << 27) | (1u << 28),
        (1u <<  3) | (1u <<  4)
    );

    const int dcol[8] = int[8](-1, -1, -1, 0, 1, 1, 1, 0);
    const int drow[8] = int[8](-1, 0, 1, 1, 1, 0, -1, -1);
    const uint dirs[8] = uint[8](DIR_NW, DIR_W, DIR_SW, DIR_S, DIR_SE, DIR_E, DIR_NE, DIR_N);

    while (frontier.x != 0u || frontier.y != 0u) {
        // Pop the lowest set bit from frontier.
        int idx;
        if (frontier.x != 0u) {
            idx = findLSB(frontier.x);
            frontier.x &= ~(1u << uint(idx));
        } else {
            int hi = findLSB(frontier.y);
            frontier.y &= ~(1u << uint(hi));
            idx = 32 + hi;
        }

        int row = idx >> 3;
        int col = idx & 7;
        int label = int(get_label(labels, row, col));

        int px = int(bx) + col - 3;
        int py = int(by) + row - 3;
        if (px < 0 || py < 0 || px >= int(img_w()) || py >= int(img_h())) continue;

        uint vmask = valence_mask(uint(px), uint(py));

        for (uint d = 0u; d < 8u; d++) {
            if ((vmask & dirs[d]) == 0u) continue;
            int nc = col + dcol[d];
            int nr = row + drow[d];
            if (nc < 0 || nc >= 8 || nr < 0 || nr >= 8) continue;
            if (get_label(labels, nr, nc) != 0u) continue;

            int npx = int(bx) + nc - 3;
            int npy = int(by) + nr - 3;
            if (npx < 0 || npy < 0 || npx >= int(img_w()) || npy >= int(img_h())) continue;

            set_label(labels, nr, nc, uint(label));
            if (label == 1) sizeA++;
            else sizeB++;

            // Set the new cell's bit in the frontier.
            int n_idx = nr * 8 + nc;
            if (n_idx < 32) frontier.x |= 1u << uint(n_idx);
            else            frontier.y |= 1u << uint(n_idx - 32);
        }
    }
}

// ---- Main ----

void main() {
    ivec2 opos = ivec2(vTexCoord * params.OutputSize.xy);
    int gx = opos.x;
    int gy = opos.y;

    // Default: passthrough from similarity graph
    vec4 src = texelFetch(Source, opos, 0);
    // Corner flags stored as float(value), read back with uint(val + 0.5)
    uint src_val = uint(src.r + 0.5);

    // Only process corner positions (both even)
    if ((gx % 2 != 0) || (gy % 2 != 0)) {
        FragColor = src;
        return;
    }

    // Corner at graph (gx, gy) corresponds to pixel block corner.
    // The 2x2 pixel block is: upper-left = (gx/2 - 1, gy/2 - 1)
    uint bx = uint(gx / 2 - 1); // upper-left pixel x
    uint by = uint(gy / 2 - 1); // upper-left pixel y

    // Border corners or corners at gx=0 / gy=0: passthrough
    if (gx <= 0 || gy <= 0) {
        FragColor = src;
        return;
    }

    // Check if this corner is within the valid interior
    // bx must be in [0, img_w-2], by in [0, img_h-2]
    if (bx >= img_w() - 1u || by >= img_h() - 1u) {
        FragColor = src;
        return;
    }

    uint flags = src_val;

    // No diagonal at this corner
    if (flags == 0u) {
        FragColor = src;
        return;
    }

    // Check fully connected: all 4 cardinal edges present
    uint top    = g(gx, gy - 1);
    uint bottom = g(gx, gy + 1);
    uint left   = g(gx - 1, gy);
    uint right  = g(gx + 1, gy);
    bool fully_connected = top != 0u && bottom != 0u && left != 0u && right != 0u;

    // Single diagonal in fully-connected block: remove it
    if (flags == 1u || flags == 2u) {
        if (fully_connected) {
            FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        } else {
            FragColor = src;
        }
        return;
    }

    // flags == 3: crossing (both diagonals present)
    if (fully_connected) {
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    int main_vote = 0;
    int anti_vote = 0;

    // Heuristic 1: Curve length
    // Main diagonal: (bx,by)-(bx+1,by+1)
    // Anti diagonal: (bx+1,by)-(bx,by+1)
    uint main_len = walk_chain(bx, by, bx + 1u, by + 1u)
                  + walk_chain(bx + 1u, by + 1u, bx, by);
    uint anti_len = walk_chain(bx + 1u, by, bx, by + 1u)
                  + walk_chain(bx, by + 1u, bx + 1u, by);
    if (main_len > anti_len) main_vote += int(main_len - anti_len);
    else if (anti_len > main_len) anti_vote += int(anti_len - main_len);

    // Heuristic 2: Sparse pixels (smaller component wins)
    uint main_size, anti_size;
    sparse_pixel_sizes(bx, by, main_size, anti_size);
    if (main_size < anti_size) main_vote += int(anti_size - main_size);
    else if (anti_size < main_size) anti_vote += int(main_size - anti_size);

    // Heuristic 3: Islands (valence-1 endpoint)
    uint v_main0 = valence_count(valence_mask(bx, by));
    uint v_main1 = valence_count(valence_mask(bx + 1u, by + 1u));
    uint v_anti0 = valence_count(valence_mask(bx + 1u, by));
    uint v_anti1 = valence_count(valence_mask(bx, by + 1u));

    if      (v_main0 == 1u) main_vote += 5;
    else if (v_main1 == 1u) main_vote += 5;
    else if (v_anti0 == 1u) anti_vote += 5;
    else if (v_anti1 == 1u) anti_vote += 5;

    // Resolve
    uint result;
    if (main_vote > anti_vote) {
        result = 1u; // keep main (backslash)
    } else if (anti_vote > main_vote) {
        result = 2u; // keep anti (slash)
    } else {
        result = 0u; // tie: remove both
    }

    // Store as float(result) — small integers stored via uintBitsToFloat become
    // denormalized floats that get flushed to zero on many GPUs.
    FragColor = vec4(float(result), 0.0, 0.0, 0.0);
}
