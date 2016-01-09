#
# Makefile
# Clay L. McLeod, 2016-01-08 19:40
#

all:
	@echo "Makefile needs your attention"

convert:
	@echo ""
	@python convert.py

reconstruct:
	@echo ""
	@python -Wignore reconstruct.py

copyffttogen:
	@echo ""
	@echo "Copying fft to gen to simulate song reconstruction"
	@cp -rf ./data/fft ./data/gen

test: clean convert copyffttogen reconstruct

.PHONY: clean
clean:
	@rm -rf ./data/tmp ./data/wav ./data/fft ./data/gen ./data/out
# vim:ft=make
#
