# CUDA GPU Puzzles — top-level build & run
#
# Common commands:
#   make p01          build & run the problem stub (fails until you fill it in)
#   make test-p01     build & run the reference solution for one puzzle
#   make test         build & run every reference solution (the test suite)
#   make build-all    compile every problem and solution without running them
#   make clean        remove build artifacts
#
# The two p04 variants:
#   make p04                raw-pointer 2D map
#   make p04_tensor_view    same puzzle using the TensorView abstraction
#   make tensor_view_intro  the standalone TensorView walkthrough from the book
#
# Override the GPU target or C++ standard on the command line, e.g.
#   make p01 ARCH=sm_90          # Hopper
#   make test STD=c++17
#
# Requirements: an NVIDIA GPU and the CUDA Toolkit (nvcc). The defaults below
# target Ampere (sm_86); set ARCH to match your card.

NVCC      ?= nvcc
ARCH      ?= sm_86
STD       ?= c++20
BUILD     ?= build
NVCCFLAGS ?= -std=$(STD) -arch=$(ARCH) -O2 -lineinfo --expt-relaxed-constexpr -Iinclude

HEADERS := $(wildcard include/*.cuh)

# Every Part I puzzle. p04 ships an extra TensorView variant.
PUZZLES := p01 p02 p03 p04 p04_tensor_view p05 p06 p07 p08

# Map a puzzle name to its source directory: p04_tensor_view -> p04, p01 -> p01.
puzzle_dir = $(firstword $(subst _, ,$(1)))

$(BUILD):
	@mkdir -p $(BUILD)

# Generate the build + run rules for a single puzzle.
define PUZZLE_RULES
$(BUILD)/$(1): problems/$(call puzzle_dir,$(1))/$(1).cu $$(HEADERS) | $$(BUILD)
	$$(NVCC) $$(NVCCFLAGS) $$< -o $$@

$(BUILD)/sol_$(1): solutions/$(call puzzle_dir,$(1))/$(1).cu $$(HEADERS) | $$(BUILD)
	$$(NVCC) $$(NVCCFLAGS) $$< -o $$@

.PHONY: $(1) test-$(1)
$(1): $(BUILD)/$(1)
	@echo "-- problems/$(1) --"
	@./$(BUILD)/$(1)

test-$(1): $(BUILD)/sol_$(1)
	@echo "-- solutions/$(1) --"
	@./$(BUILD)/sol_$(1)
endef

$(foreach p,$(PUZZLES),$(eval $(call PUZZLE_RULES,$(p))))

.PHONY: test build-all tensor_view_intro clean help

# Run every reference solution; stops at the first failure.
test: $(addprefix test-,$(PUZZLES))
	@echo "All solutions passed."

# Compile everything (handy for a quick "does it still build?" check).
build-all: $(addprefix $(BUILD)/,$(PUZZLES)) $(addprefix $(BUILD)/sol_,$(PUZZLES))

# The TensorView introduction program that lives alongside the book chapter.
tensor_view_intro: book/src/puzzle_04/intro.cu $(HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $(BUILD)/tensor_view_intro
	@./$(BUILD)/tensor_view_intro

clean:
	rm -rf $(BUILD)

help:
	@echo "make p01            run a problem stub"
	@echo "make test-p01       run one reference solution"
	@echo "make test           run all reference solutions"
	@echo "make build-all      compile everything"
	@echo "make clean          remove build/"
