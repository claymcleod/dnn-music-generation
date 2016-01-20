#
# Makefile
# Clay L. McLeod, 2016-01-08 19:40
#

# Set to either 'cpu' or 'gpu', based on your configuration
device=cpu

default: convert train generate reconstruct

convert:
	@echo ""
	@echo "###########"
	@echo "# Convert #"
	@echo "###########"
	@echo ""
	@THEANO_FLAGS=mode=FAST_RUN,device=${device},floatX=float32 python -Wignore convert.py

train:
	@echo ""
	@echo "#########"
	@echo "# Train #"
	@echo "#########"
	@echo ""
	@THEANO_FLAGS=mode=FAST_RUN,device=${device},floatX=float32 python -Wignore train.py

generate:
	@echo ""
	@echo "############"
	@echo "# Generate #"
	@echo "############"
	@echo ""
	@THEANO_FLAGS=mode=FAST_RUN,device=${device},floatX=float32 python -Wignore generate.py

reconstruct:
	@echo ""
	@echo "###############"
	@echo "# Reconstruct #"
	@echo "###############"
	@echo ""
	@THEANO_FLAGS=mode=FAST_RUN,device=${device},floatX=float32 python -Wignore reconstruct.py

spectrogram:
	@python -Wignore spectrogram.py

copyffttogen:
	@echo ""
	@echo "Copying fft to gen to simulate song reconstruction"
	@cp -rf ./data/fft ./data/gen

test: clean convert copyffttogen reconstruct

.PHONY: clean
clean:
	@rm -rf ./data/*/
