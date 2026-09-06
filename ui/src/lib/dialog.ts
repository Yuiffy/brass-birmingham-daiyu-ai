interface DialogOptions {
	onClose: () => void;
	onKeydown?: (event: KeyboardEvent) => void;
}

export function modalDialog(node: HTMLDialogElement, options: DialogOptions) {
	const previousFocus = document.activeElement;
	const previousOverflow = document.body.style.overflow;
	document.body.style.overflow = 'hidden';
	node.showModal();

	const cancel = (event: Event) => {
		event.preventDefault();
		options.onClose();
	};
	const click = (event: MouseEvent) => {
		if (event.target === node) options.onClose();
	};
	const keydown = (event: KeyboardEvent) => options.onKeydown?.(event);
	node.addEventListener('cancel', cancel);
	node.addEventListener('click', click);
	node.addEventListener('keydown', keydown);

	return {
		update(nextOptions: DialogOptions) { options = nextOptions; },
		destroy() {
			node.removeEventListener('cancel', cancel);
			node.removeEventListener('click', click);
			node.removeEventListener('keydown', keydown);
			node.close();
			document.body.style.overflow = previousOverflow;
			if (previousFocus instanceof HTMLElement && previousFocus.isConnected) {
				previousFocus.focus({ preventScroll: true });
			}
		}
	};
}
