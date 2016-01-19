#
# Makefile
# Clay L. McLeod, 2016-01-08 19:40
#

convert:
	@echo ""
	@python -Wignore convert.py

generate:
	@echo ""
	@python -Wignore generate.py

reconstruct:
	@echo ""
	@python -Wignore reconstruct.py

train:
	@echo ""
	@python -Wignore train.py

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
