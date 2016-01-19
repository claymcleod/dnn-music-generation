#
# Makefile
# Clay L. McLeod, 2016-01-08 19:40
#

default: convert train generate reconstruct

convert:
	@echo "# Running 'Convert'"
	@python -Wignore convert.py
	@rm -rf ./data/tmp

train:
	@echo "# Running 'Train'"
	@python -Wignore train.py

generate:
	@echo "# Running 'Generate'"
	@python -Wignore generate.py

reconstruct:
	@echo "# Running 'Reconstruct'"
	@python -Wignore reconstruct.py

copyffttogen:
	@echo ""
	@echo "Copying fft to gen to simulate song reconstruction"
	@cp -rf ./data/fft ./data/gen

test: clean convert copyffttogen reconstruct

.PHONY: clean
clean:
	@rm -rf ./data/tmp ./data/wav ./data/fft ./data/gen ./data/out ./data/plot ./data/weights

spectrogram:
	@python -Wignore spectrogram.py
# vim:ft=make
#
