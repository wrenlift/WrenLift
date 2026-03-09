/*
 * embed.c — Example of embedding WrenLift in a C application.
 *
 * Build (macOS):
 *   cargo build --release
 *   cc -o embed examples/embed.c -Iexamples \
 *      -Ltarget/release -lwren_lift \
 *      -framework Security -framework CoreFoundation
 *   DYLD_LIBRARY_PATH=target/release ./embed
 *
 * Build (Linux):
 *   cargo build --release
 *   cc -o embed examples/embed.c -Iexamples \
 *      -Ltarget/release -lwren_lift -lm -ldl -lpthread
 *   LD_LIBRARY_PATH=target/release ./embed
 *
 * Static linking (any platform):
 *   cc -o embed examples/embed.c -Iexamples \
 *      target/release/libwren_lift.a -lm -ldl -lpthread
 */
#include <stdio.h>
#include <string.h>
#include "wrenlift.h"

/* ---------- Callbacks ---------- */

static void writeFn(WrenVM* vm, const char* text) {
    (void)vm;
    printf("%s", text);
}

static void errorFn(WrenVM* vm, WrenErrorType type,
                     const char* module, int line, const char* msg) {
    (void)vm;
    switch (type) {
        case WREN_ERROR_COMPILE:
            fprintf(stderr, "[%s line %d] Compile error: %s\n", module, line, msg);
            break;
        case WREN_ERROR_RUNTIME:
            fprintf(stderr, "Runtime error: %s\n", msg);
            break;
        case WREN_ERROR_STACK_TRACE:
            fprintf(stderr, "  [%s line %d] in %s\n", module, line, msg);
            break;
    }
}

/* ---------- 1. Hello World ---------- */

static void demo_hello(WrenVM* vm) {
    printf("--- Hello World ---\n");
    wrenInterpret(vm, "hello", "System.print(\"Hello from WrenLift!\")");
    printf("\n");
}

/* ---------- 2. Calling Wren methods from C ---------- */

static void demo_call_wren(WrenVM* vm) {
    printf("--- Calling Wren from C ---\n");

    const char* src =
        "class Math {\n"
        "  static add(a, b) { return a + b }\n"
        "  static mul(a, b) { return a * b }\n"
        "  static clamp(x, lo, hi) {\n"
        "    if (x < lo) return lo\n"
        "    if (x > hi) return hi\n"
        "    return x\n"
        "  }\n"
        "}\n";

    if (wrenInterpret(vm, "math", src) != WREN_RESULT_SUCCESS) return;

    wrenEnsureSlots(vm, 4);

    /* Math.add(100, 200) */
    WrenHandle* add = wrenMakeCallHandle(vm, "add(_,_)");
    wrenGetVariable(vm, "math", "Math", 0);
    wrenSetSlotDouble(vm, 1, 100.0);
    wrenSetSlotDouble(vm, 2, 200.0);
    if (wrenCall(vm, add) == WREN_RESULT_SUCCESS) {
        printf("  add(100, 200) = %.0f\n", wrenGetSlotDouble(vm, 0));
    }

    /* Math.mul(6, 7) */
    WrenHandle* mul = wrenMakeCallHandle(vm, "mul(_,_)");
    wrenGetVariable(vm, "math", "Math", 0);
    wrenSetSlotDouble(vm, 1, 6.0);
    wrenSetSlotDouble(vm, 2, 7.0);
    if (wrenCall(vm, mul) == WREN_RESULT_SUCCESS) {
        printf("  mul(6, 7)     = %.0f\n", wrenGetSlotDouble(vm, 0));
    }

    /* Math.clamp(42, 0, 10) */
    WrenHandle* clamp = wrenMakeCallHandle(vm, "clamp(_,_,_)");
    wrenGetVariable(vm, "math", "Math", 0);
    wrenSetSlotDouble(vm, 1, 42.0);
    wrenSetSlotDouble(vm, 2, 0.0);
    wrenSetSlotDouble(vm, 3, 10.0);
    if (wrenCall(vm, clamp) == WREN_RESULT_SUCCESS) {
        printf("  clamp(42,0,10)= %.0f\n", wrenGetSlotDouble(vm, 0));
    }

    wrenReleaseHandle(vm, add);
    wrenReleaseHandle(vm, mul);
    wrenReleaseHandle(vm, clamp);
    printf("\n");
}

/* ---------- 3. Passing strings between C and Wren ---------- */

static void demo_strings(WrenVM* vm) {
    printf("--- Strings ---\n");

    const char* src =
        "class Greeter {\n"
        "  static greet(name) { return \"Hello, \" + name + \"!\" }\n"
        "}\n";

    if (wrenInterpret(vm, "greeter", src) != WREN_RESULT_SUCCESS) return;

    wrenEnsureSlots(vm, 2);
    wrenGetVariable(vm, "greeter", "Greeter", 0);

    WrenHandle* greet = wrenMakeCallHandle(vm, "greet(_)");

    const char* names[] = {"Alice", "Bob", "World"};
    for (int i = 0; i < 3; i++) {
        wrenGetVariable(vm, "greeter", "Greeter", 0);
        wrenSetSlotString(vm, 1, names[i]);
        if (wrenCall(vm, greet) == WREN_RESULT_SUCCESS) {
            printf("  %s\n", wrenGetSlotString(vm, 0));
        }
    }

    wrenReleaseHandle(vm, greet);
    printf("\n");
}

/* ---------- 4. Building lists from C ---------- */

static void demo_lists(WrenVM* vm) {
    printf("--- Lists ---\n");

    wrenEnsureSlots(vm, 3);
    wrenSetSlotNewList(vm, 0);

    /* Build a list of numbers from C */
    for (int i = 1; i <= 5; i++) {
        wrenSetSlotDouble(vm, 1, (double)(i * i));
        wrenInsertInList(vm, 0, -1, 1);
    }

    int count = wrenGetListCount(vm, 0);
    printf("  Squares: [");
    for (int i = 0; i < count; i++) {
        wrenGetListElement(vm, 0, i, 2);
        if (i > 0) printf(", ");
        printf("%.0f", wrenGetSlotDouble(vm, 2));
    }
    printf("]\n");

    /* Modify an element */
    wrenSetSlotDouble(vm, 1, 999.0);
    wrenSetListElement(vm, 0, 2, 1);
    wrenGetListElement(vm, 0, 2, 2);
    printf("  After set [2]=999: third element is %.0f\n", wrenGetSlotDouble(vm, 2));
    printf("\n");
}

/* ---------- 5. Building maps from C ---------- */

static void demo_maps(WrenVM* vm) {
    printf("--- Maps ---\n");

    wrenEnsureSlots(vm, 3);
    wrenSetSlotNewMap(vm, 0);

    /* Populate a map */
    wrenSetSlotString(vm, 1, "name");
    wrenSetSlotString(vm, 2, "WrenLift");
    wrenSetMapValue(vm, 0, 1, 2);

    wrenSetSlotString(vm, 1, "version");
    wrenSetSlotString(vm, 2, "0.5.0");
    wrenSetMapValue(vm, 0, 1, 2);

    wrenSetSlotString(vm, 1, "fast");
    wrenSetSlotBool(vm, 2, true);
    wrenSetMapValue(vm, 0, 1, 2);

    printf("  Map has %d entries\n", wrenGetMapCount(vm, 0));

    /* Look up a value by key */
    wrenSetSlotString(vm, 1, "name");
    wrenGetMapValue(vm, 0, 1, 2);
    printf("  map[\"name\"]    = \"%s\"\n", wrenGetSlotString(vm, 2));

    wrenSetSlotString(vm, 1, "version");
    wrenGetMapValue(vm, 0, 1, 2);
    printf("  map[\"version\"] = \"%s\"\n", wrenGetSlotString(vm, 2));

    /* Check containment */
    wrenSetSlotString(vm, 1, "fast");
    printf("  contains \"fast\": %s\n",
           wrenGetMapContainsKey(vm, 0, 1) ? "yes" : "no");
    wrenSetSlotString(vm, 1, "missing");
    printf("  contains \"missing\": %s\n",
           wrenGetMapContainsKey(vm, 0, 1) ? "yes" : "no");

    /* Remove a key */
    wrenEnsureSlots(vm, 4);
    wrenSetSlotString(vm, 1, "fast");
    wrenRemoveMapValue(vm, 0, 1, 3);
    printf("  Removed \"fast\" (was %s), map now has %d entries\n",
           wrenGetSlotBool(vm, 3) ? "true" : "false",
           wrenGetMapCount(vm, 0));
    printf("\n");
}

/* ---------- 6. Multiple modules ---------- */

static void demo_modules(WrenVM* vm) {
    printf("--- Modules ---\n");

    wrenInterpret(vm, "config", "var Width = 1920\nvar Height = 1080");
    wrenInterpret(vm, "game",   "var Title = \"My Game\"");

    wrenEnsureSlots(vm, 1);

    wrenGetVariable(vm, "config", "Width", 0);
    double w = wrenGetSlotDouble(vm, 0);

    wrenGetVariable(vm, "config", "Height", 0);
    double h = wrenGetSlotDouble(vm, 0);

    wrenGetVariable(vm, "game", "Title", 0);
    const char* title = wrenGetSlotString(vm, 0);

    printf("  %s at %.0fx%.0f\n", title, w, h);
    printf("  Has 'config' module: %s\n", wrenHasModule(vm, "config") ? "yes" : "no");
    printf("  Has 'audio' module:  %s\n", wrenHasModule(vm, "audio")  ? "yes" : "no");

    /* Variables are isolated per module */
    printf("  'config' has 'Width':  %s\n",
           wrenHasVariable(vm, "config", "Width") ? "yes" : "no");
    printf("  'config' has 'Title':  %s\n",
           wrenHasVariable(vm, "config", "Title") ? "yes" : "no");
    printf("\n");
}

/* ---------- 7. Error handling ---------- */

static void demo_errors(WrenVM* vm) {
    printf("--- Error Handling ---\n");
    fflush(stdout);

    /* Runtime error */
    WrenInterpretResult r = wrenInterpret(vm, "err",
        "Fiber.abort(\"something went wrong\")");
    fflush(stderr);
    printf("  Result: %s\n",
           r == WREN_RESULT_RUNTIME_ERROR ? "RUNTIME_ERROR (expected)" : "unexpected");
    printf("\n");
}

/* ---------- Main ---------- */

int main(void) {
    /* Force line-buffered stdout so output order is predictable */
    setvbuf(stdout, NULL, _IOLBF, 0);

    printf("WrenLift C Embedding Example (API v%d)\n\n", wrenGetVersionNumber());

    WrenConfiguration config;
    wrenInitConfiguration(&config);
    config.writeFn = writeFn;
    config.errorFn = errorFn;

    WrenVM* vm = wrenNewVM(&config);

    demo_hello(vm);
    demo_call_wren(vm);
    demo_strings(vm);
    demo_lists(vm);
    demo_maps(vm);
    demo_modules(vm);
    demo_errors(vm);

    wrenCollectGarbage(vm);
    wrenFreeVM(vm);

    printf("Done.\n");
    return 0;
}
